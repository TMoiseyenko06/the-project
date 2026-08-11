"""Resume support: a manifest of source images that have already been processed.

An entry is keyed by the content of the source image *plus* the settings that
affect the result (prompt, model, generation kwargs). Re-running the same batch
with the same prompt skips finished images; changing the prompt or the model
reprocesses them, because the key changes.

Only entries with ``status == "ok"`` whose output file still exists are treated
as done — failures are retried on the next run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .storage import atomic_write_json

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_HASH_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: Path) -> str:
    """Content hash of a file, read in chunks so large images stay cheap."""
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@dataclass
class ManifestEntry:
    source: str  # original path, for reporting
    source_hash: str
    output: str | None  # path relative to the output root, e.g. "unsorted/x.png"
    prompt: str
    status: str  # "ok" | "failed"
    timestamp: float
    duration: float = 0.0
    error: str | None = None
    attempts: int = 1


class Manifest:
    """Thread-safe, atomically-persisted record of processed source images."""

    def __init__(self, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None):
        self.path = Path(path)
        self.root = Path(root) if root else self.path.parent
        self._lock = threading.RLock()
        self._entries: dict[str, ManifestEntry] = {}
        self._dirty = False
        self.load()

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self._entries = {}
            if not self.path.is_file():
                return
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                # A manifest truncated by a hard crash must not block a rerun:
                # move it aside and start fresh rather than raising.
                backup = self.path.with_suffix(f".corrupt.{int(time.time())}.json")
                log.warning("Manifest %s is corrupt (%s); moved to %s", self.path, exc, backup)
                self.path.rename(backup)
                return
            for key, raw in (data.get("entries") or {}).items():
                try:
                    self._entries[key] = ManifestEntry(**raw)
                except TypeError:
                    log.warning("Dropping malformed manifest entry %s", key)
            log.info("Loaded manifest with %d entries", len(self._entries))

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": SCHEMA_VERSION,
                "updated_at": time.time(),
                "entries": {k: asdict(v) for k, v in self._entries.items()},
            }
            atomic_write_json(self.path, payload)
            self._dirty = False

    def save_if_dirty(self) -> None:
        with self._lock:
            if self._dirty:
                self.save()

    # -- keys -------------------------------------------------------------
    @staticmethod
    def make_key(source_hash: str, prompt: str, signature: dict[str, Any]) -> str:
        material = "|".join([source_hash, prompt, _stable_json(signature)])
        return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()

    # -- queries ----------------------------------------------------------
    def _locate(self, entry: ManifestEntry) -> Path | None:
        """Find an entry's output, following it if it was moved into an album.

        Organising images in the UI moves files between directories, so the
        recorded path goes stale. Rather than treating that as "missing" and
        reprocessing the source, look for the same filename elsewhere in the
        output tree and heal the entry.
        """
        if not entry.output:
            return None
        recorded = self.root / entry.output
        if recorded.is_file():
            return recorded
        basename = Path(entry.output).name
        candidates = [self.root / "unsorted"]
        candidates += sorted(p for p in self.root.glob("album_*") if p.is_dir())
        for directory in candidates:
            moved = directory / basename
            if moved.is_file():
                try:
                    entry.output = moved.relative_to(self.root).as_posix()
                except ValueError:
                    entry.output = moved.as_posix()
                self._dirty = True
                return moved
        return None

    def is_done(self, key: str) -> bool:
        """True when this exact work was completed and its output still exists."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.status != "ok" or not entry.output:
                return False
            return self._locate(entry) is not None

    def get(self, key: str) -> ManifestEntry | None:
        with self._lock:
            return self._entries.get(key)

    def entries(self) -> dict[str, ManifestEntry]:
        with self._lock:
            return dict(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- mutations --------------------------------------------------------
    def record_success(self, key: str, source: Path, source_hash: str, output: Path,
                       prompt: str, duration: float) -> ManifestEntry:
        try:
            relative = output.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            relative = output.as_posix()
        with self._lock:
            previous = self._entries.get(key)
            entry = ManifestEntry(
                source=str(source),
                source_hash=source_hash,
                output=relative,
                prompt=prompt,
                status="ok",
                timestamp=time.time(),
                duration=duration,
                error=None,
                attempts=(previous.attempts + 1) if previous else 1,
            )
            self._entries[key] = entry
            self._dirty = True
            return entry

    def record_failure(self, key: str, source: Path, source_hash: str, prompt: str,
                       error: str) -> ManifestEntry:
        with self._lock:
            previous = self._entries.get(key)
            entry = ManifestEntry(
                source=str(source),
                source_hash=source_hash,
                output=None,
                prompt=prompt,
                status="failed",
                timestamp=time.time(),
                error=error[:2000],
                attempts=(previous.attempts + 1) if previous else 1,
            )
            self._entries[key] = entry
            self._dirty = True
            return entry

    def prune_missing(self) -> int:
        """Drop successful entries whose output is gone (deleted in the UI).

        Moved-into-an-album outputs are relocated rather than pruned.
        """
        removed = 0
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry.status == "ok" and entry.output and self._locate(entry) is None:
                    del self._entries[key]
                    removed += 1
            if removed:
                self._dirty = True
        return removed
