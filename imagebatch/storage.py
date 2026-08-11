"""Output storage and album organisation.

Layout under ``output_dir``::

    outputs/
      albums.json        # source of truth for album membership + nesting
      manifest.json      # processed source images (see manifest.py)
      unsorted/          # freshly processed images land here by default
      album_<slug>/      # one folder per album, flat regardless of nesting
      logs/              # per-run json logs
      .staging/          # extracted zip uploads / thumbnails / prepared zips

``albums.json`` is authoritative for both membership and hierarchy. Album
nesting is recorded as a ``parent`` slug rather than mirrored as nested
directories: re-parenting an album is then a single atomic JSON update instead
of a recursive directory move that could be interrupted half-done. Zip exports
do rebuild the hierarchy, so downloads come out nested.
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
SCHEMA_VERSION = 2  # v1 had no `parent` field; migrated transparently on read
MAX_DEPTH = 12  # guard against pathological nesting


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


def safe_component(name: str) -> str:
    """Sanitise an album name for use as a single path segment inside a zip."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip().strip(".")
    return cleaned[:80] or "album"


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


def unique_dir(parent: Path, base: str) -> Path:
    """Create and return a non-colliding directory ``parent/base`` (or `base_N`).

    Used for per-run staging folders keyed by a timestamp, where two calls in
    the same wall-clock second must not be allowed to merge into one folder.
    """
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / base
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    for i in range(1, 10_000):
        candidate = parent / f"{base}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise StorageError(f"could not find a free directory name for {base} in {parent}")


@dataclass
class Album:
    slug: str
    name: str
    created_at: float
    images: list[str]
    parent: str | None = None

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
        self._migrate(data)
        return data

    @staticmethod
    def _migrate(data: dict[str, Any]) -> None:
        """Bring an older albums.json up to the current schema in memory.

        v1 files have no ``parent`` key; those albums become top level. A parent
        pointing at a missing album would orphan it, so that is repaired too.
        """
        albums = data["albums"]
        for meta in albums.values():
            meta.setdefault("parent", None)
        for slug, meta in albums.items():
            parent = meta.get("parent")
            if parent is not None and parent not in albums:
                log.warning("Album %s referenced missing parent %s; moving to top level",
                            slug, parent)
                meta["parent"] = None
        # A cycle would make the tree walk non-terminating; break any that exist.
        for slug in albums:
            seen, cursor = {slug}, albums[slug].get("parent")
            while cursor is not None:
                if cursor in seen:
                    log.warning("Album cycle detected at %s; moving it to top level", slug)
                    albums[slug]["parent"] = None
                    break
                seen.add(cursor)
                cursor = albums.get(cursor, {}).get("parent")

    def _write(self, data: dict[str, Any]) -> None:
        data["version"] = SCHEMA_VERSION
        atomic_write_json(self.albums_file, data)

    @staticmethod
    def _album_from(slug: str, meta: dict[str, Any]) -> Album:
        return Album(slug=slug, name=meta.get("name", slug),
                     created_at=meta.get("created_at", 0.0),
                     images=list(meta.get("images", [])),
                     parent=meta.get("parent"))

    # -- hierarchy helpers ------------------------------------------------
    @staticmethod
    def _children_of(data: dict[str, Any], parent: str | None) -> list[str]:
        slugs = [s for s, m in data["albums"].items() if m.get("parent") == parent]
        slugs.sort(key=lambda s: data["albums"][s].get("name", s).lower())
        return slugs

    @classmethod
    def _descendants_of(cls, data: dict[str, Any], slug: str) -> list[str]:
        found: list[str] = []
        stack = list(cls._children_of(data, slug))
        while stack:
            current = stack.pop(0)
            found.append(current)
            stack = list(cls._children_of(data, current)) + stack
        return found

    @staticmethod
    def _ancestors_of(data: dict[str, Any], slug: str) -> list[str]:
        chain: list[str] = []
        cursor = data["albums"].get(slug, {}).get("parent")
        while cursor is not None and cursor in data["albums"] and cursor not in chain:
            chain.append(cursor)
            cursor = data["albums"][cursor].get("parent")
        return chain

    @classmethod
    def _depth_of(cls, data: dict[str, Any], slug: str | None) -> int:
        if slug is None:
            return 0
        return len(cls._ancestors_of(data, slug)) + 1

    @classmethod
    def _unique_sibling_name(cls, data: dict[str, Any], parent: str | None,
                             name: str, exclude: str | None = None) -> str:
        """Return *name*, suffixed if a sibling already uses it.

        Used where a rename is a side effect (promoting children of a deleted
        album) rather than something the user typed — those paths get an error.
        """
        taken = {
            data["albums"][s].get("name", "").strip().lower()
            for s in cls._children_of(data, parent) if s != exclude
        }
        if name.strip().lower() not in taken:
            return name
        for i in range(2, 1000):
            candidate = f"{name} ({i})"
            if candidate.strip().lower() not in taken:
                return candidate
        raise StorageError(f"could not find a free name near {name!r}")

    def _check_sibling_name(self, data: dict[str, Any], parent: str | None,
                            name: str, exclude: str | None = None) -> None:
        for sibling in self._children_of(data, parent):
            if sibling == exclude:
                continue
            if data["albums"][sibling].get("name", "").strip().lower() == name.strip().lower():
                where = "at the top level" if parent is None else \
                    f"inside {data['albums'][parent].get('name', parent)!r}"
                raise StorageError(f"an album named {name!r} already exists {where}")

    # -- queries ----------------------------------------------------------
    def list_albums(self) -> list[Album]:
        """All albums, sorted by name (flat). See :meth:`album_tree` for order."""
        with self._lock:
            data = self._read()
        albums = [self._album_from(slug, meta) for slug, meta in data["albums"].items()]
        albums.sort(key=lambda a: a.name.lower())
        return albums

    def album_tree(self) -> list[tuple[Album, int]]:
        """Albums in depth-first display order, paired with their depth (0 = top)."""
        with self._lock:
            data = self._read()
        ordered: list[tuple[Album, int]] = []

        def walk(parent: str | None, depth: int) -> None:
            for slug in self._children_of(data, parent):
                ordered.append((self._album_from(slug, data["albums"][slug]), depth))
                walk(slug, depth + 1)

        walk(None, 0)
        return ordered

    def get_album(self, slug: str) -> Album:
        with self._lock:
            data = self._read()
        meta = data["albums"].get(slug)
        if meta is None:
            raise StorageError(f"no such album: {slug}")
        return self._album_from(slug, meta)

    def children(self, slug: str | None) -> list[Album]:
        with self._lock:
            data = self._read()
        return [self._album_from(s, data["albums"][s]) for s in self._children_of(data, slug)]

    def descendants(self, slug: str) -> list[Album]:
        with self._lock:
            data = self._read()
        return [self._album_from(s, data["albums"][s]) for s in self._descendants_of(data, slug)]

    def ancestors(self, slug: str) -> list[Album]:
        with self._lock:
            data = self._read()
        return [self._album_from(s, data["albums"][s]) for s in self._ancestors_of(data, slug)]

    def path_name(self, slug: str, separator: str = " / ") -> str:
        """Human-readable full path, e.g. ``"Keepers / Portraits"``."""
        if slug == UNSORTED:
            return UNSORTED_LABEL
        with self._lock:
            data = self._read()
        if slug not in data["albums"]:
            return slug
        parts = [data["albums"][s].get("name", s) for s in
                 reversed(self._ancestors_of(data, slug))]
        parts.append(data["albums"][slug].get("name", slug))
        return separator.join(parts)

    def find_by_name(self, name: str, parent: str | None = None) -> Album | None:
        """Find an album by name among the children of *parent*."""
        target = name.strip().lower()
        with self._lock:
            data = self._read()
        for slug in self._children_of(data, parent):
            if data["albums"][slug].get("name", "").strip().lower() == target:
                return self._album_from(slug, data["albums"][slug])
        return None

    def list_unsorted(self) -> list[str]:
        """Filenames in ``unsorted/``, read from disk (it has no JSON record)."""
        if not self.unsorted_dir.is_dir():
            return []
        names = [p.name for p in self.unsorted_dir.iterdir() if p.is_file() and is_image(p)]
        names.sort()
        return names

    def list_images(self, album: str | None,
                    include_descendants: bool = False) -> list[str]:
        """Filenames in an album (or unsorted), filtered to files that exist.

        With *include_descendants* the result spans nested albums; because two
        sub-albums may hold the same filename, use :meth:`list_images_located`
        when you need to know which album each one came from.
        """
        if album is None or album == UNSORTED:
            return self.list_unsorted()
        if not include_descendants:
            directory = self.album_dir(album)
            return [name for name in self.get_album(album).images
                    if (directory / name).is_file()]
        return [name for _, name in self.list_images_located(album, include_descendants=True)]

    def list_images_located(self, album: str | None,
                            include_descendants: bool = False) -> list[tuple[str, str]]:
        """``(album_slug, filename)`` pairs, optionally spanning sub-albums."""
        if album is None or album == UNSORTED:
            return [(UNSORTED, name) for name in self.list_unsorted()]
        with self._lock:
            data = self._read()
            if album not in data["albums"]:
                raise StorageError(f"no such album: {album}")
            slugs = [album]
            if include_descendants:
                slugs += self._descendants_of(data, album)
            located: list[tuple[str, str]] = []
            for slug in slugs:
                directory = self.album_dir(slug)
                for name in data["albums"][slug].get("images", []):
                    if (directory / name).is_file():
                        located.append((slug, name))
        return located

    def count_images(self, album: str | None, include_descendants: bool = False) -> int:
        if album is None or album == UNSORTED:
            return len(self.list_unsorted())
        return len(self.list_images_located(album, include_descendants=include_descendants))

    def total_counts(self) -> dict[str, int]:
        """Direct image count per album (plus unsorted), excluding sub-albums."""
        counts = {UNSORTED: len(self.list_unsorted())}
        for album in self.list_albums():
            counts[album.slug] = len(self.list_images(album.slug))
        return counts

    # -- mutations --------------------------------------------------------
    def create_album(self, name: str, parent: str | None = None) -> Album:
        name = name.strip()
        if not name:
            raise StorageError("album name cannot be empty")
        if parent == UNSORTED:
            parent = None  # Unsorted is not a real album, so it can't be a parent
        with self._lock:
            data = self._read()
            if parent is not None and parent not in data["albums"]:
                raise StorageError(f"no such parent album: {parent}")
            if parent is not None and self._depth_of(data, parent) >= MAX_DEPTH:
                raise StorageError(f"albums cannot be nested more than {MAX_DEPTH} deep")
            self._check_sibling_name(data, parent, name)

            slug = slugify(name)
            base_slug, i = slug, 1
            while slug in data["albums"]:
                slug = f"{base_slug}-{i}"
                i += 1
            created = time.time()
            data["albums"][slug] = {
                "name": name,
                "created_at": created,
                "images": [],
                "parent": parent,
            }
            self.album_dir(slug).mkdir(parents=True, exist_ok=True)
            self._write(data)
        log.info("Created album %r (slug=%s, parent=%s)", name, slug, parent)
        return Album(slug=slug, name=name, created_at=created, images=[], parent=parent)

    def rename_album(self, slug: str, new_name: str) -> Album:
        """Rename an album. The slug and directory stay put so links survive."""
        new_name = new_name.strip()
        if not new_name:
            raise StorageError("album name cannot be empty")
        with self._lock:
            data = self._read()
            if slug not in data["albums"]:
                raise StorageError(f"no such album: {slug}")
            self._check_sibling_name(data, data["albums"][slug].get("parent"),
                                     new_name, exclude=slug)
            data["albums"][slug]["name"] = new_name
            self._write(data)
        log.info("Renamed album %s -> %r", slug, new_name)
        return self.get_album(slug)

    def move_album(self, slug: str, new_parent: str | None) -> Album:
        """Re-parent an album (pass ``None`` to move it to the top level)."""
        if new_parent == UNSORTED:
            new_parent = None
        with self._lock:
            data = self._read()
            if slug not in data["albums"]:
                raise StorageError(f"no such album: {slug}")
            if new_parent is not None:
                if new_parent not in data["albums"]:
                    raise StorageError(f"no such parent album: {new_parent}")
                if new_parent == slug:
                    raise StorageError("an album cannot be inside itself")
                if slug in self._ancestors_of(data, new_parent):
                    # Allowing this would detach the subtree into a cycle.
                    raise StorageError(
                        f"cannot move {data['albums'][slug].get('name', slug)!r} into its "
                        "own sub-album"
                    )
                subtree_depth = 1 + max(
                    (self._depth_of(data, d) - self._depth_of(data, slug)
                     for d in self._descendants_of(data, slug)), default=0)
                if self._depth_of(data, new_parent) + subtree_depth > MAX_DEPTH:
                    raise StorageError(
                        f"that move would nest albums more than {MAX_DEPTH} deep")
            if data["albums"][slug].get("parent") == new_parent:
                return self._album_from(slug, data["albums"][slug])
            self._check_sibling_name(data, new_parent,
                                     data["albums"][slug].get("name", slug), exclude=slug)
            data["albums"][slug]["parent"] = new_parent
            self._write(data)
        log.info("Moved album %s under %s", slug, new_parent or "<top level>")
        return self.get_album(slug)

    def delete_album(self, slug: str, delete_images: bool = False,
                     recursive: bool = False) -> dict[str, int]:
        """Delete an album.

        Images are moved back to ``unsorted/`` unless *delete_images* is set.
        Sub-albums are promoted to the deleted album's parent unless *recursive*
        is set, in which case the whole subtree goes too.

        Returns ``{"images": n, "albums": n}``.
        """
        with self._lock:
            data = self._read()
            if slug not in data["albums"]:
                raise StorageError(f"no such album: {slug}")
            parent = data["albums"][slug].get("parent")
            children = self._children_of(data, slug)

            if recursive:
                targets = [slug] + self._descendants_of(data, slug)
            else:
                targets = [slug]
                for child in children:
                    # Promoting can collide with an existing name at the
                    # destination; this is a side effect, so resolve it silently.
                    data["albums"][child]["name"] = self._unique_sibling_name(
                        data, parent, data["albums"][child].get("name", child),
                        exclude=child)
                    data["albums"][child]["parent"] = parent

            affected = 0
            for target in targets:
                directory = self.album_dir(target)
                if directory.is_dir():
                    for path in list(directory.iterdir()):
                        if not path.is_file():
                            continue
                        if delete_images:
                            path.unlink()
                        else:
                            shutil.move(str(path),
                                        str(unique_path(self.unsorted_dir, path.name)))
                        affected += 1
                    shutil.rmtree(directory, ignore_errors=True)
                del data["albums"][target]
            self._write(data)

        log.info("Deleted %d album(s) rooted at %s (%d images %s)", len(targets), slug,
                 affected, "deleted" if delete_images else "moved to unsorted")
        return {"images": affected, "albums": len(targets)}

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

    def register_image(self, album: str | None, filename: str) -> None:
        """Record a file already written into an album's directory.

        Used by the batch runner when results are sent straight to an album
        instead of landing in ``unsorted/``.
        """
        key = album or UNSORTED
        if key == UNSORTED:
            return  # unsorted is read from disk, so there is nothing to record
        name = Path(filename).name
        with self._lock:
            data = self._read()
            if key not in data["albums"]:
                raise StorageError(f"no such album: {key}")
            images = data["albums"][key]["images"]
            if name not in images:
                images.append(name)
            self._write(data)

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

    def zip_album(self, album: str | None, dest_dir: Path | None = None,
                  include_descendants: bool = True) -> Path:
        """Zip an album (or unsorted) and return the archive path.

        Sub-albums are written as nested folders inside the archive, so the
        hierarchy survives the download even though it is flat on disk.
        """
        key = album or UNSORTED
        if key == UNSORTED:
            base_name = UNSORTED
            entries = [(UNSORTED, name) for name in self.list_unsorted()]
            prefixes = {UNSORTED: ""}
        else:
            with self._lock:
                data = self._read()
                if key not in data["albums"]:
                    raise StorageError(f"no such album: {key}")
                base_name = slugify(data["albums"][key].get("name", key))
                # Walk down from the root, building each album's path as we go.
                prefixes = {key: ""}
                queue = [key]
                while queue:
                    current = queue.pop(0)
                    if not include_descendants:
                        break
                    for child in self._children_of(data, current):
                        parent_prefix = prefixes[current]
                        child_name = safe_component(data["albums"][child].get("name", child))
                        prefixes[child] = (f"{parent_prefix}/{child_name}"
                                           if parent_prefix else child_name)
                        queue.append(child)
            entries = self.list_images_located(key, include_descendants=include_descendants)

        if not entries:
            raise StorageError(f"album {base_name!r} has no images to download")

        dest_dir = dest_dir or (self.staging_dir / "downloads")
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"{base_name}_{int(time.time())}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for slug, filename in entries:
                path = self.dir_for(slug) / filename
                if not path.is_file():
                    continue
                prefix = prefixes.get(slug, "")
                zf.write(path, arcname=f"{prefix}/{filename}" if prefix else filename)
        log.info("Wrote %s (%d images)", archive, len(entries))
        return archive
