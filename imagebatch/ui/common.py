"""Small helpers shared by the UI tabs."""

from __future__ import annotations

from pathlib import Path

from ..storage import UNSORTED, UNSORTED_LABEL, AlbumStore
from ..thumbnails import ThumbnailCache


def album_choices(store: AlbumStore, include_unsorted: bool = True) -> list[tuple[str, str]]:
    """Dropdown choices as ``(label, value)``; values are slugs (or "unsorted")."""
    choices: list[tuple[str, str]] = []
    if include_unsorted:
        choices.append((f"{UNSORTED_LABEL} ({len(store.list_unsorted())})", UNSORTED))
    for album in store.list_albums():
        choices.append((f"{album.name} ({len(store.list_images(album.slug))})", album.slug))
    return choices


def album_label(store: AlbumStore, value: str) -> str:
    if value == UNSORTED:
        return UNSORTED_LABEL
    try:
        return store.get_album(value).name
    except Exception:  # noqa: BLE001 - the album may have just been deleted
        return value


def page_count(total: int, page_size: int) -> int:
    if page_size <= 0:
        return 1
    return max(1, (total + page_size - 1) // page_size)


def page_slice(names: list[str], page: int, page_size: int) -> list[str]:
    """Return the names on *page* (1-indexed), clamped to the valid range."""
    if page_size <= 0:
        return names
    pages = page_count(len(names), page_size)
    page = min(max(1, page), pages)
    start = (page - 1) * page_size
    return names[start:start + page_size]


def gallery_items(store: AlbumStore, album: str, names: list[str],
                  thumbnails: ThumbnailCache, selected: set[str]) -> list[tuple[str, str]]:
    """Build ``(thumbnail_path, caption)`` pairs, marking selected images."""
    directory = store.dir_for(album)
    items: list[tuple[str, str]] = []
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        caption = f"✅ {name}" if name in selected else name
        items.append((thumbnails.get(path), caption))
    return items


def full_path(store: AlbumStore, album: str, name: str) -> str | None:
    path = Path(store.dir_for(album)) / Path(name).name
    return str(path) if path.is_file() else None
