from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from imagebatch.storage import UNSORTED, AlbumStore, StorageError, slugify, unique_path
from tests.conftest import make_image


def test_layout_created(store: AlbumStore) -> None:
    assert store.unsorted_dir.is_dir()
    assert store.albums_file.is_file()
    assert json.loads(store.albums_file.read_text())["albums"] == {}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("My Best Shots", "my-best-shots"),
        ("  spaced   out  ", "spaced-out"),
        ("Rejects!!! 2024", "rejects-2024"),
        ("a/b\\c", "a-b-c"),
    ],
)
def test_slugify(name: str, expected: str) -> None:
    assert slugify(name) == expected


def test_slugify_non_ascii_is_stable() -> None:
    slug = slugify("日本語")
    assert slug and "/" not in slug
    assert slug == slugify("日本語")


def test_create_and_list_albums(store: AlbumStore) -> None:
    album = store.create_album("Keepers")
    assert album.slug == "keepers"
    assert store.album_dir("keepers").is_dir()
    assert [a.name for a in store.list_albums()] == ["Keepers"]

    with pytest.raises(StorageError):
        store.create_album("keepers")  # case-insensitive duplicate


def test_create_album_slug_collision(store: AlbumStore) -> None:
    a = store.create_album("Best Of")
    b = store.create_album("best-of!")
    assert a.slug != b.slug
    assert store.album_dir(b.slug).is_dir()


def test_assign_moves_file_and_updates_json(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Keepers")

    moved = store.assign(["a.png"], None, album.slug)

    assert moved == ["a.png"]
    assert not (store.unsorted_dir / "a.png").exists()
    assert (store.album_dir(album.slug) / "a.png").is_file()
    assert store.get_album(album.slug).images == ["a.png"]
    assert store.list_unsorted() == []


def test_assign_between_albums(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    first = store.create_album("First")
    second = store.create_album("Second")

    store.assign(["a.png"], None, first.slug)
    store.assign(["a.png"], first.slug, second.slug)

    assert store.get_album(first.slug).images == []
    assert store.get_album(second.slug).images == ["a.png"]
    assert (store.album_dir(second.slug) / "a.png").is_file()


def test_assign_handles_name_collision(store: AlbumStore) -> None:
    """A later run can produce a second output with the same filename."""
    album = store.create_album("Keepers")
    make_image(store.unsorted_dir / "a.png", color="blue")
    store.assign(["a.png"], None, album.slug)
    make_image(store.unsorted_dir / "a.png", color="red")

    moved = store.assign(["a.png"], None, album.slug)

    assert moved == ["a_1.png"]
    assert sorted(store.get_album(album.slug).images) == ["a.png", "a_1.png"]
    assert store.list_unsorted() == []


def test_assign_back_to_unsorted(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Keepers")
    store.assign(["a.png"], None, album.slug)

    store.assign(["a.png"], album.slug, None)

    assert store.list_unsorted() == ["a.png"]
    assert store.get_album(album.slug).images == []


def test_assign_ignores_path_traversal(store: AlbumStore) -> None:
    secret = store.root.parent / "secret.png"
    make_image(secret)
    album = store.create_album("Keepers")

    moved = store.assign(["../secret.png"], None, album.slug)

    assert moved == []
    assert secret.is_file()


def test_assign_missing_file_is_skipped(store: AlbumStore) -> None:
    album = store.create_album("Keepers")
    assert store.assign(["ghost.png"], None, album.slug) == []


def test_delete_album_moves_images_to_unsorted(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Temp")
    store.assign(["a.png"], None, album.slug)

    affected = store.delete_album(album.slug)

    assert affected == {"images": 1, "albums": 1}
    assert store.list_unsorted() == ["a.png"]
    assert not store.album_dir(album.slug).exists()
    assert store.list_albums() == []


def test_delete_album_with_images(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Temp")
    store.assign(["a.png"], None, album.slug)

    store.delete_album(album.slug, delete_images=True)

    assert store.list_unsorted() == []
    assert store.list_albums() == []


def test_rename_album_keeps_images(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Old")
    store.assign(["a.png"], None, album.slug)

    renamed = store.rename_album(album.slug, "New")

    assert renamed.name == "New"
    assert renamed.slug == album.slug
    assert store.list_images(album.slug) == ["a.png"]

    other = store.create_album("Another")
    with pytest.raises(StorageError):
        store.rename_album(other.slug, "New")


def test_delete_images(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    make_image(store.unsorted_dir / "b.png")

    assert store.delete_images(["a.png"], None) == 1
    assert store.list_unsorted() == ["b.png"]


def test_list_images_filters_missing_files(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Keepers")
    store.assign(["a.png"], None, album.slug)
    (store.album_dir(album.slug) / "a.png").unlink()

    assert store.list_images(album.slug) == []
    assert store.get_album(album.slug).images == ["a.png"]  # json untouched until sync


def test_sync_reconciles_disk_and_json(store: AlbumStore) -> None:
    album = store.create_album("Keepers")
    make_image(store.album_dir(album.slug) / "stray.png")
    store.assign([], None, album.slug)
    data = json.loads(store.albums_file.read_text())
    data["albums"][album.slug]["images"] = ["ghost.png"]
    store.albums_file.write_text(json.dumps(data))

    report = store.sync()

    assert report["added"] == [f"{album.slug}/stray.png"]
    assert report["removed"] == [f"{album.slug}/ghost.png"]
    assert store.get_album(album.slug).images == ["stray.png"]


def test_zip_album(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    make_image(store.unsorted_dir / "b.png")
    album = store.create_album("Keepers")
    store.assign(["a.png", "b.png"], None, album.slug)

    archive = store.zip_album(album.slug)

    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["a.png", "b.png"]


def test_zip_empty_album_raises(store: AlbumStore) -> None:
    album = store.create_album("Empty")
    with pytest.raises(StorageError):
        store.zip_album(album.slug)


def test_total_counts(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    make_image(store.unsorted_dir / "b.png")
    album = store.create_album("Keepers")
    store.assign(["a.png"], None, album.slug)

    counts = store.total_counts()

    assert counts[UNSORTED] == 1
    assert counts[album.slug] == 1


def test_unique_path(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png")
    assert unique_path(tmp_path, "a.png").name == "a_1.png"
    assert unique_path(tmp_path, "b.png").name == "b.png"


def test_store_survives_reopen(tmp_path: Path) -> None:
    store = AlbumStore(tmp_path / "outputs")
    album = store.create_album("Keepers")
    make_image(store.unsorted_dir / "a.png")
    store.assign(["a.png"], None, album.slug)

    reopened = AlbumStore(tmp_path / "outputs")

    assert [a.name for a in reopened.list_albums()] == ["Keepers"]
    assert reopened.list_images(album.slug) == ["a.png"]
