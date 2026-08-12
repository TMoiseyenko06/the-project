"""Cross-cutting tags, layered on top of AlbumStore.

Where an album is a folder an image physically lives in, a tag is metadata
attached to an image wherever it lives: a category (e.g. "person") with one or
more values (e.g. "Alex", "Jordan" — a photo can have two people in it) per
image. An image can carry tags in several categories at once, independent of
which album it's filed into.

Images are addressed as ``"<album_or_unsorted>/<filename>"``, the same
addressing the gallery already uses for tokens. Because images move between
albums, :class:`~imagebatch.storage.AlbumStore` calls :meth:`TagStore.rename_image`
/ :meth:`TagStore.forget_image` at the same points it moves or deletes files —
keeping tags in sync as part of that same operation, rather than a separate
reconciliation pass that could drift or race.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterable

from .storage import UNSORTED, StorageError, atomic_write_json

log = logging.getLogger(__name__)

TAGS_FILE = "tags.json"
SCHEMA_VERSION = 1
UNTAGGED = "(untagged)"  # bucket label for a group_by category with no value


def image_key(album: str | None, filename: str) -> str:
    """The same "album/filename" addressing the gallery already uses for tokens."""
    return f"{album or UNSORTED}/{Path(filename).name}"


def split_key(key: str) -> tuple[str, str]:
    album, _, name = key.partition("/")
    return album, name


class TagStore:
    """Thread-safe category / value / per-image tag assignment bookkeeping."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        if not self.path.exists():
            atomic_write_json(self.path, {"version": SCHEMA_VERSION, "categories": {},
                                          "assignments": {}})

    # -- raw json -----------------------------------------------------------
    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": SCHEMA_VERSION, "categories": {}, "assignments": {}}
        except json.JSONDecodeError as exc:
            raise StorageError(f"{self.path} is corrupt: {exc}") from exc
        data.setdefault("categories", {})
        data.setdefault("assignments", {})
        return data

    def _write(self, data: dict) -> None:
        data["version"] = SCHEMA_VERSION
        atomic_write_json(self.path, data)

    @staticmethod
    def _slug(name: str) -> str:
        return name.strip().lower()

    # -- categories -----------------------------------------------------------
    def list_categories(self) -> list[dict]:
        """[{"key", "name", "values"}, ...] sorted by name."""
        with self._lock:
            data = self._read()
        items = [{"key": key, "name": meta["name"], "values": list(meta["values"])}
                 for key, meta in data["categories"].items()]
        items.sort(key=lambda c: c["name"].lower())
        return items

    def create_category(self, name: str) -> str:
        name = name.strip()
        if not name:
            raise StorageError("category name cannot be empty")
        key = self._slug(name)
        if not key:
            raise StorageError(f"invalid category name: {name!r}")
        with self._lock:
            data = self._read()
            if key in data["categories"]:
                raise StorageError(f"a category named {name!r} already exists")
            data["categories"][key] = {"name": name, "values": []}
            self._write(data)
        log.info("Created tag category %r", name)
        return key

    def rename_category(self, key: str, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise StorageError("category name cannot be empty")
        with self._lock:
            data = self._read()
            if key not in data["categories"]:
                raise StorageError(f"no such category: {key}")
            data["categories"][key]["name"] = new_name
            self._write(data)
        log.info("Renamed tag category %s -> %r", key, new_name)

    def delete_category(self, key: str) -> int:
        """Delete a category and strip it out of every image's tags.

        Returns the number of images that had this category removed.
        """
        with self._lock:
            data = self._read()
            if key not in data["categories"]:
                raise StorageError(f"no such category: {key}")
            del data["categories"][key]
            affected = 0
            for tags in data["assignments"].values():
                if key in tags:
                    del tags[key]
                    affected += 1
            data["assignments"] = {k: v for k, v in data["assignments"].items() if v}
            self._write(data)
        log.info("Deleted tag category %s (%d image(s) affected)", key, affected)
        return affected

    def category_name(self, key: str) -> str:
        with self._lock:
            data = self._read()
        meta = data["categories"].get(key)
        return meta["name"] if meta else key

    def list_values(self, category: str) -> list[str]:
        with self._lock:
            data = self._read()
        meta = data["categories"].get(category)
        if meta is None:
            raise StorageError(f"no such category: {category}")
        return list(meta["values"])

    # -- per-image tags -------------------------------------------------------
    def get_tags(self, album: str | None, filename: str) -> dict[str, list[str]]:
        """``{category_key: [values]}`` for one image, empty dict if untagged."""
        with self._lock:
            data = self._read()
        return {k: list(v) for k, v in
                data["assignments"].get(image_key(album, filename), {}).items()}

    def set_tags(self, album: str | None, filename: str, category: str,
                values: Iterable[str]) -> list[str]:
        """Replace an image's values for one category; an empty list clears it.

        New values are added to the category's known-values list automatically
        (auto-vivified), so tagging doesn't require a separate "define this
        value first" step. Returns the stored (deduped, sorted) values.
        """
        cleaned = sorted({v.strip() for v in values if v.strip()})
        key = image_key(album, filename)
        with self._lock:
            data = self._read()
            if category not in data["categories"]:
                raise StorageError(f"no such category: {category}")
            known = data["categories"][category]["values"]
            for value in cleaned:
                if value not in known:
                    known.append(value)
            entry = data["assignments"].setdefault(key, {})
            if cleaned:
                entry[category] = cleaned
            else:
                entry.pop(category, None)
            if not entry:
                data["assignments"].pop(key, None)
            self._write(data)
        return cleaned

    def bulk_add(self, images: Iterable[tuple[str | None, str]], category: str,
                value: str) -> int:
        """Add one value to a category for many images at once (additive, doesn't
        clear other values already on those images). Returns the count updated."""
        value = value.strip()
        if not value:
            raise StorageError("tag value cannot be empty")
        with self._lock:
            data = self._read()
            if category not in data["categories"]:
                raise StorageError(f"no such category: {category}")
            known = data["categories"][category]["values"]
            if value not in known:
                known.append(value)
            updated = 0
            for album, filename in images:
                key = image_key(album, filename)
                entry = data["assignments"].setdefault(key, {})
                current = entry.setdefault(category, [])
                if value not in current:
                    current.append(value)
                    current.sort()
                updated += 1
            self._write(data)
        return updated

    # -- sync hooks, called by AlbumStore on move/delete -----------------------
    def rename_image(self, old_album: str | None, old_name: str,
                     new_album: str | None, new_name: str) -> None:
        old_key, new_key = image_key(old_album, old_name), image_key(new_album, new_name)
        if old_key == new_key:
            return
        with self._lock:
            data = self._read()
            if old_key in data["assignments"]:
                data["assignments"][new_key] = data["assignments"].pop(old_key)
                self._write(data)

    def forget_image(self, album: str | None, filename: str) -> None:
        key = image_key(album, filename)
        with self._lock:
            data = self._read()
            if key in data["assignments"]:
                del data["assignments"][key]
                self._write(data)

    # -- search -----------------------------------------------------------------
    def search(self, category: str, value: str,
              group_by: str | None = None) -> dict[str, list[tuple[str, str]]]:
        """Images tagged ``category=value``, bucketed by ``group_by``.

        Returns ``{bucket_label: [(album, filename), ...]}``. Without
        ``group_by`` everything lands in a single ``""`` bucket. An image
        matching the search but missing a value in ``group_by`` lands in the
        :data:`UNTAGGED` bucket rather than silently disappearing. An image
        with multiple values in ``group_by`` appears once per value — e.g. a
        photo tagged with two people shows up under both people's buckets.
        """
        with self._lock:
            data = self._read()
        if category not in data["categories"]:
            raise StorageError(f"no such category: {category}")
        if group_by is not None and group_by not in data["categories"]:
            raise StorageError(f"no such category: {group_by}")

        buckets: dict[str, list[tuple[str, str]]] = {}
        for key, tags in data["assignments"].items():
            if value not in tags.get(category, []):
                continue
            album, filename = split_key(key)
            if group_by is None:
                buckets.setdefault("", []).append((album, filename))
                continue
            group_values = tags.get(group_by) or [UNTAGGED]
            for group_value in group_values:
                buckets.setdefault(group_value, []).append((album, filename))

        for entries in buckets.values():
            entries.sort()
        return buckets

    def all_tagged_images(self) -> dict[str, dict[str, list[str]]]:
        """Every tagged image and its tags, keyed by ``album/filename``."""
        with self._lock:
            data = self._read()
        return {k: {c: list(vs) for c, vs in tags.items()}
                for k, tags in data["assignments"].items()}
