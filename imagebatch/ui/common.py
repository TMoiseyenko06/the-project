"""Small helpers shared by the UI tabs."""

from __future__ import annotations

from pathlib import Path

from ..storage import UNSORTED, UNSORTED_LABEL, AlbumStore


INDENT = " "  # em space: dropdown labels collapse ordinary leading spaces


def album_choices(store: AlbumStore, include_unsorted: bool = True,
                  include_top_level: bool = False,
                  exclude: set[str] | None = None) -> list[tuple[str, str]]:
    """Dropdown choices as ``(label, value)``; values are slugs (or "unsorted").

    Albums are listed in tree order and indented by depth. *include_top_level*
    adds a "(top level)" entry with value ``None`` for parent pickers, and
    *exclude* drops slugs that would be invalid choices (an album cannot be
    moved inside itself or its own descendants).
    """
    exclude = exclude or set()
    choices: list[tuple[str, str]] = []
    if include_unsorted:
        choices.append((f"{UNSORTED_LABEL} ({len(store.list_unsorted())})", UNSORTED))
    if include_top_level:
        choices.append(("— top level —", TOP_LEVEL))
    for album, depth in store.album_tree():
        if album.slug in exclude:
            continue
        direct = len(store.list_images(album.slug))
        nested = store.count_images(album.slug, include_descendants=True)
        count = f"{direct}" if nested == direct else f"{direct} / {nested}"
        choices.append((f"{INDENT * depth}{album.name} ({count})", album.slug))
    return choices


# Sentinel used as a dropdown value for "no parent". Gradio cannot round-trip a
# real None through a Dropdown value, so it travels as this string instead.
TOP_LEVEL = "__top__"


def parent_value(value: str | None) -> str | None:
    """Translate a parent dropdown value into what the store expects."""
    if value in (None, "", TOP_LEVEL, UNSORTED):
        return None
    return value


def album_label(store: AlbumStore, value: str, full_path: bool = False) -> str:
    if value == UNSORTED:
        return UNSORTED_LABEL
    try:
        return store.path_name(value) if full_path else store.get_album(value).name
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


def full_path(store: AlbumStore, album: str, name: str) -> str | None:
    path = Path(store.dir_for(album)) / Path(name).name
    return str(path) if path.is_file() else None
