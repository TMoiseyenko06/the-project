"""Output storage and album organisation.

Layout under ``output_dir``::

    outputs/
      albums.json        # source of truth for album membership
      manifest.json      # processed source images (see manifest.py)
      unsorted/          # freshly processed images land here
      album_<slug>/      # one folder per album
      logs/              # per-run jsonl logs
      .staging/          # extracted zip uploads / scratch

``albums.json`` is authoritative for membership; every membership change is a
file move plus a JSON update, performed under a lock and written atomically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

UNSORTED = "unsorted"
UNSORTED_LABEL = "Unsorted"
ALBUM_DIR_PREFIX = "album_"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALBUMS_FILE = "albums.json"
SCHEMA_VERSION = 1


class StorageError(RuntimeError):
    """Raised for recoverable, user-facing storage problems."""


def slugify(name: str) -> str:
    """Turn an album name into a filesystem-safe slug."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        # Names made entirely of non-ASCII characters still need a stable slug.
        slug = "album-" + str(abs(hash(name)) % 10_000_000)
    return slug[:64]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def unique_path(directory: Path, filename: str) -> Path:
    """Return a non-colliding path inside *directory* for *filename*."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for i in range(1, 10_000):
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise StorageError(f"could not find a free filename for {filename} in {directory}")


@dataclass
class Album:
    slug: str
    name: str
    created_at: float
    images: list[str]

    @property
    def count(self) -> int:
        return len(self.images)


class AlbumStore:
    """Thread-safe album/image bookkeeping on top of a directory tree."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.unsorted_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if not self.albums_file.exists():
            atomic_write_json(self.albums_file, {"version": SCHEMA_VERSION, "albums": {}})

    # -- paths ------------------------------------------------------------
    @property
    def albums_file(self) -> Path:
        return self.root / ALBUMS_FILE

    @property
    def unsorted_dir(self) -> Path:
        return self.root / UNSORTED

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def staging_dir(self) -> Path:
        return self.root / ".staging"

    def album_dir(self, slug: str) -> Path:
        return self.root / f"{ALBUM_DIR_PREFIX}{slug}"

    def dir_for(self, album: str | None) -> Path:
        """Directory for an album slug, or the unsorted directory for None."""
        if album is None or album == UNSORTED:
            return self.unsorted_dir
        return self.album_dir(album)

    def image_path(self, album: str | None, filename: str) -> Path:
        safe = Path(filename).name  # never let a caller escape the album dir
        return self.dir_for(album) / safe

    # -- raw json ---------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.albums_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": SCHEMA_VERSION, "albums": {}}
        except json.JSONDecodeError as exc:
            raise StorageError(f"{self.albums_file} is corrupt: {exc}") from exc
        data.setdefault("albums", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        data["version"] = SCHEMA_VERSION
        atomic_write_json(self.albums_file, data)

    # -- queries ----------------------------------------------------------
    def list_albums(self) -> list[Album]:
        with self._lock:
            data = self._read()
        albums = [
            Album(slug=slug, name=meta.get("name", slug),
                  created_at=meta.get("created_at", 0.0),
                  images=list(meta.get("images", [])))
            for slug, meta in data["albums"].items()
        ]
        albums.sort(key=lambda a: a.name.lower())
        return albums

    def get_album(self, slug: str) -> Album:
        with self._lock:
            data = self._read()
        meta = data["albums"].get(slug)
        if meta is None:
            raise StorageError(f"no such album: {slug}")
        return Album(slug=slug, name=meta.get("name", slug),
                     created_at=meta.get("created_at", 0.0),
                     images=list(meta.get("images", [])))

    def find_by_name(self, name: str) -> Album | None:
        target = name.strip().lower()
        for album in self.list_albums():
            if album.name.strip().lower() == target:
                return album
        return None

    def list_unsorted(self) -> list[str]:
        """Filenames in ``unsorted/``, read from disk (it has no JSON record)."""
        if not self.unsorted_dir.is_dir():
            return []
        names = [p.name for p in self.unsorted_dir.iterdir() if p.is_file() and is_image(p)]
        names.sort()
        return names

    def list_images(self, album: str | None) -> list[str]:
        """Filenames in an album (or unsorted), filtered to files that exist."""
        if album is None or album == UNSORTED:
            return self.list_unsorted()
        album_obj = self.get_album(album)
        directory = self.album_dir(album)
        return [name for name in album_obj.images if (directory / name).is_file()]

    def total_counts(self) -> dict[str, int]:
        counts = {UNSORTED: len(self.list_unsorted())}
        for album in self.list_albums():
            counts[album.slug] = len(self.list_images(album.slug))
        return counts

    # -- mutations --------------------------------------------------------
    def create_album(self, name: str) -> Album:
        name = name.strip()
        if not name:
            raise StorageError("album name cannot be empty")
        with self._lock:
            data = self._read()
            existing = {meta.get("name", "").strip().lower(): slug
                        for slug, meta in data["albums"].items()}
            if name.lower() in existing:
                raise StorageError(f"an album named {name!r} already exists")
            slug = slugify(name)
            base_slug, i = slug, 1
            while slug in data["albums"]:
                slug = f"{base_slug}-{i}"
                i += 1
            data["albums"][slug] = {
                "name": name,
                "created_at": time.time(),
                "images": [],
            }
            self.album_dir(slug).mkdir(parents=True, exist_ok=True)
            self._write(data)
        log.info("Created album %r (slug=%s)", name, slug)
        return Album(slug=slug, name=name, created_at=time.time(), images=[])

    def rename_album(self, slug: str, new_name: str) -> Album:
        """Rename an album. The slug and directory stay put so links survive."""
        new_name = new_name.strip()
        if not new_name:
            raise StorageError("album name cannot be empty")
        with self._lock:
            data = self._read()
            if slug not in data["albums"]:
                raise StorageError(f"no such album: {slug}")
            for other, meta in data["albums"].items():
                if other != slug and meta.get("name", "").strip().lower() == new_name.lower():
                    raise StorageError(f"an album named {new_name!r} already exists")
            data["albums"][slug]["name"] = new_name
            self._write(data)
        log.info("Renamed album %s -> %r", slug, new_name)
        return self.get_album(slug)

    def delete_album(self, slug: str, delete_images: bool = False) -> int:
        """Delete an album.

        By default the images are moved back to ``unsorted/`` so nothing is lost;
        pass ``delete_images=True`` to remove the files too. Returns the number
        of images moved or deleted.
        """
        with self._lock:
            data = self._read()
            if slug not in data["albums"]:
                raise StorageError(f"no such album: {slug}")
            directory = self.album_dir(slug)
            affected = 0
            if directory.is_dir():
                for path in list(directory.iterdir()):
                    if not path.is_file():
                        continue
                    if delete_images:
                        path.unlink()
                    else:
                        shutil.move(str(path), str(unique_path(self.unsorted_dir, path.name)))
                    affected += 1
                shutil.rmtree(directory, ignore_errors=True)
            del data["albums"][slug]
            self._write(data)
        log.info("Deleted album %s (%d images %s)", slug, affected,
                 "deleted" if delete_images else "moved to unsorted")
        return affected

    def assign(self, filenames: Iterable[str], source_album: str | None,
               target_album: str | None) -> list[str]:
        """Move images from one album (or unsorted) into another.

        Returns the resulting filenames in the target, which may differ from the
        inputs when a name collision forced a rename.
        """
        filenames = [Path(f).name for f in filenames]
        if not filenames:
            return []
        source_key = source_album or UNSORTED
        target_key = target_album or UNSORTED
        if source_key == target_key:
            return filenames

        with self._lock:
            data = self._read()
            for key in (source_key, target_key):
                if key != UNSORTED and key not in data["albums"]:
                    raise StorageError(f"no such album: {key}")

            source_dir = self.dir_for(source_album)
            target_dir = self.dir_for(target_album)
            target_dir.mkdir(parents=True, exist_ok=True)

            moved: list[str] = []
            for name in filenames:
                src = source_dir / name
                if not src.is_file():
                    log.warning("Skipping %s: not found in %s", name, source_dir)
                    continue
                dst = unique_path(target_dir, name)
                shutil.move(str(src), str(dst))
                moved.append(dst.name)
                if source_key != UNSORTED:
                    images = data["albums"][source_key]["images"]
                    if name in images:
                        images.remove(name)
                if target_key != UNSORTED:
                    data["albums"][target_key]["images"].append(dst.name)
            self._write(data)

        log.info("Moved %d image(s) %s -> %s", len(moved), source_key, target_key)
        return moved

    def delete_images(self, filenames: Iterable[str], album: str | None) -> int:
        """Permanently delete images from an album (or unsorted)."""
        filenames = [Path(f).name for f in filenames]
        key = album or UNSORTED
        removed = 0
        with self._lock:
            data = self._read()
            directory = self.dir_for(album)
            for name in filenames:
                path = directory / name
                if path.is_file():
                    path.unlink()
                    removed += 1
                if key != UNSORTED and name in data["albums"].get(key, {}).get("images", []):
                    data["albums"][key]["images"].remove(name)
            self._write(data)
        return removed

    def register_output(self, path: Path) -> str:
        """Record a freshly written file in ``unsorted/`` (no JSON entry needed)."""
        if path.parent != self.unsorted_dir:
            raise StorageError(f"outputs must land in {self.unsorted_dir}, got {path}")
        return path.name

    # -- maintenance ------------------------------------------------------
    def sync(self) -> dict[str, list[str]]:
        """Reconcile ``albums.json`` with what is actually on disk.

        Files present in an album directory but missing from the JSON are added;
        entries whose files have vanished are dropped. Returns a report of the
        changes so the UI can surface them.
        """
        report: dict[str, list[str]] = {"added": [], "removed": []}
        with self._lock:
            data = self._read()
            for slug, meta in data["albums"].items():
                directory = self.album_dir(slug)
                directory.mkdir(parents=True, exist_ok=True)
                on_disk = {p.name for p in directory.iterdir() if p.is_file() and is_image(p)}
                recorded = list(meta.get("images", []))
                kept = [name for name in recorded if name in on_disk]
                for name in recorded:
                    if name not in on_disk:
                        report["removed"].append(f"{slug}/{name}")
                for name in sorted(on_disk - set(kept)):
                    kept.append(name)
                    report["added"].append(f"{slug}/{name}")
                meta["images"] = kept
            self._write(data)
        if report["added"] or report["removed"]:
            log.info("Sync: +%d -%d entries", len(report["added"]), len(report["removed"]))
        return report

    def zip_album(self, album: str | None, dest_dir: Path | None = None) -> Path:
        """Zip an album (or unsorted) and return the archive path."""
        key = album or UNSORTED
        name = UNSORTED if key == UNSORTED else slugify(self.get_album(key).name)
        directory = self.dir_for(album)
        images = self.list_images(album)
        if not images:
            raise StorageError(f"album {name!r} has no images to download")
        dest_dir = dest_dir or (self.staging_dir / "downloads")
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"{name}_{int(time.time())}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in images:
                path = directory / filename
                if path.is_file():
                    zf.write(path, arcname=filename)
        log.info("Wrote %s (%d images)", archive, len(images))
        return archive
