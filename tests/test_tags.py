"""TagStore unit tests, plus AlbumStore integration (move/delete keeping tags in sync)."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagebatch.storage import UNSORTED, AlbumStore, StorageError
from imagebatch.tags import TagStore, UNTAGGED, image_key, split_key
from tests.conftest import make_image


@pytest.fixture
def tag_store(tmp_path: Path) -> TagStore:
    return TagStore(tmp_path / "tags.json")


# -- addressing -------------------------------------------------------------
def test_image_key_shape() -> None:
    assert image_key("keepers", "a.png") == "keepers/a.png"
    assert image_key(None, "a.png") == "unsorted/a.png"
    assert image_key(UNSORTED, "a.png") == "unsorted/a.png"


def test_split_key_roundtrip() -> None:
    assert split_key(image_key("keepers", "a.png")) == ("keepers", "a.png")


# -- categories ---------------------------------------------------------
def test_create_and_list_categories(tag_store: TagStore) -> None:
    tag_store.create_category("Person")
    tag_store.create_category("Pose")

    categories = tag_store.list_categories()

    assert [c["name"] for c in categories] == ["Person", "Pose"]
    assert all(c["values"] == [] for c in categories)


def test_create_category_slug_is_lowercase(tag_store: TagStore) -> None:
    key = tag_store.create_category("Person")
    assert key == "person"


def test_duplicate_category_rejected(tag_store: TagStore) -> None:
    tag_store.create_category("Person")
    with pytest.raises(StorageError, match="already exists"):
        tag_store.create_category("person")


def test_empty_category_name_rejected(tag_store: TagStore) -> None:
    with pytest.raises(StorageError):
        tag_store.create_category("   ")


def test_rename_category(tag_store: TagStore) -> None:
    key = tag_store.create_category("Person")
    tag_store.rename_category(key, "People")
    assert tag_store.category_name(key) == "People"


def test_rename_missing_category_rejected(tag_store: TagStore) -> None:
    with pytest.raises(StorageError, match="no such category"):
        tag_store.rename_category("ghost", "New")


def test_delete_category_strips_from_images(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.create_category("pose")
    tag_store.set_tags(None, "a.png", "person", ["Alex"])
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])

    affected = tag_store.delete_category("person")

    assert affected == 1
    assert tag_store.get_tags(None, "a.png") == {"pose": ["Sitting"]}
    assert tag_store.list_categories() == [
        {"key": "pose", "name": "pose", "values": ["Sitting"]}
    ]


def test_delete_category_removes_empty_assignment(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "person", ["Alex"])

    tag_store.delete_category("person")

    assert tag_store.get_tags(None, "a.png") == {}


# -- per-image tagging ----------------------------------------------------
def test_set_and_get_tags(tag_store: TagStore) -> None:
    tag_store.create_category("person")

    tag_store.set_tags(None, "a.png", "person", ["Alex"])

    assert tag_store.get_tags(None, "a.png") == {"person": ["Alex"]}


def test_set_tags_auto_vivifies_values(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "person", ["Alex", "Jordan"])
    assert tag_store.list_values("person") == ["Alex", "Jordan"]


def test_set_tags_dedupes_and_sorts(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    stored = tag_store.set_tags(None, "a.png", "person", ["Jordan", "alex ", "Jordan"])
    assert stored == ["Jordan", "alex"]  # sorted lexicographically, dupes/whitespace gone


def test_set_tags_empty_list_clears_category(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "person", ["Alex"])

    tag_store.set_tags(None, "a.png", "person", [])

    assert tag_store.get_tags(None, "a.png") == {}


def test_set_tags_replaces_not_merges(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "person", ["Alex"])
    tag_store.set_tags(None, "a.png", "person", ["Jordan"])
    assert tag_store.get_tags(None, "a.png") == {"person": ["Jordan"]}


def test_set_tags_unknown_category_rejected(tag_store: TagStore) -> None:
    with pytest.raises(StorageError, match="no such category"):
        tag_store.set_tags(None, "a.png", "ghost", ["x"])


def test_untagged_image_returns_empty(tag_store: TagStore) -> None:
    assert tag_store.get_tags(None, "nope.png") == {}


def test_multiple_categories_on_one_image(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.create_category("pose")
    tag_store.set_tags(None, "a.png", "person", ["Alex", "Jordan"])
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])

    assert tag_store.get_tags(None, "a.png") == {
        "person": ["Alex", "Jordan"], "pose": ["Sitting"],
    }


# -- bulk tagging (batch runs) --------------------------------------------
def test_bulk_add_tags_many_images(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    images = [(None, "a.png"), (None, "b.png"), ("keepers", "c.png")]

    updated = tag_store.bulk_add(images, "pose", "Standing")

    assert updated == 3
    for album, name in images:
        assert tag_store.get_tags(album, name) == {"pose": ["Standing"]}


def test_bulk_add_is_additive(tag_store: TagStore) -> None:
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "person", ["Alex"])

    tag_store.bulk_add([(None, "a.png")], "person", "Jordan")

    assert tag_store.get_tags(None, "a.png") == {"person": ["Alex", "Jordan"]}


def test_bulk_add_empty_value_rejected(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    with pytest.raises(StorageError):
        tag_store.bulk_add([(None, "a.png")], "pose", "  ")


# -- search / group-by ----------------------------------------------------
def test_search_flat(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])
    tag_store.set_tags(None, "b.png", "pose", ["Standing"])

    result = tag_store.search("pose", "Sitting")

    assert result == {"": [(UNSORTED, "a.png")]}


def test_search_grouped_by_other_category(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])
    tag_store.set_tags(None, "a.png", "person", ["Alex"])
    tag_store.set_tags(None, "b.png", "pose", ["Sitting"])
    tag_store.set_tags(None, "b.png", "person", ["Jordan"])
    tag_store.set_tags(None, "c.png", "pose", ["Standing"])
    tag_store.set_tags(None, "c.png", "person", ["Alex"])

    result = tag_store.search("pose", "Sitting", group_by="person")

    assert result == {
        "Alex": [(UNSORTED, "a.png")],
        "Jordan": [(UNSORTED, "b.png")],
    }


def test_search_is_symmetric(tag_store: TagStore) -> None:
    """Searching person=Alex grouped by pose is the "vice versa" the user wants."""
    tag_store.create_category("pose")
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])
    tag_store.set_tags(None, "a.png", "person", ["Alex"])
    tag_store.set_tags(None, "b.png", "pose", ["Standing"])
    tag_store.set_tags(None, "b.png", "person", ["Alex"])

    result = tag_store.search("person", "Alex", group_by="pose")

    assert result == {
        "Sitting": [(UNSORTED, "a.png")],
        "Standing": [(UNSORTED, "b.png")],
    }


def test_search_multi_value_image_appears_in_each_group(tag_store: TagStore) -> None:
    """A photo of two people shows up under both people when grouping by person."""
    tag_store.create_category("pose")
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])
    tag_store.set_tags(None, "a.png", "person", ["Alex", "Jordan"])

    result = tag_store.search("pose", "Sitting", group_by="person")

    assert result == {
        "Alex": [(UNSORTED, "a.png")],
        "Jordan": [(UNSORTED, "a.png")],
    }


def test_search_untagged_group_bucket(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    tag_store.create_category("person")
    tag_store.set_tags(None, "a.png", "pose", ["Sitting"])  # no person set

    result = tag_store.search("pose", "Sitting", group_by="person")

    assert result == {UNTAGGED: [(UNSORTED, "a.png")]}


def test_search_no_matches_returns_empty(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    assert tag_store.search("pose", "Nonexistent") == {}


def test_search_unknown_category_rejected(tag_store: TagStore) -> None:
    with pytest.raises(StorageError, match="no such category"):
        tag_store.search("ghost", "x")


def test_search_unknown_group_by_rejected(tag_store: TagStore) -> None:
    tag_store.create_category("pose")
    with pytest.raises(StorageError, match="no such category"):
        tag_store.search("pose", "Sitting", group_by="ghost")


# -- persistence ------------------------------------------------------------
def test_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "tags.json"
    store = TagStore(path)
    store.create_category("person")
    store.set_tags(None, "a.png", "person", ["Alex"])

    reopened = TagStore(path)

    assert reopened.get_tags(None, "a.png") == {"person": ["Alex"]}
    assert reopened.list_values("person") == ["Alex"]


# -- AlbumStore integration: tags follow images on move/delete ------------
def test_assign_carries_tags_to_new_location(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    store.tags.create_category("person")
    store.tags.set_tags(None, "a.png", "person", ["Alex"])
    album = store.create_album("Keepers")

    store.assign(["a.png"], None, album.slug)

    assert store.tags.get_tags(None, "a.png") == {}
    assert store.tags.get_tags(album.slug, "a.png") == {"person": ["Alex"]}


def test_assign_carries_tags_through_rename_collision(store: AlbumStore) -> None:
    """If the destination filename gets renamed to avoid a collision, tags follow it."""
    album = store.create_album("Keepers")
    make_image(store.album_dir(album.slug) / "a.png", color="blue")
    store.tags.create_category("person")

    make_image(store.unsorted_dir / "a.png", color="red")
    store.tags.set_tags(None, "a.png", "person", ["Alex"])
    moved = store.assign(["a.png"], None, album.slug)

    assert moved == ["a_1.png"]
    assert store.tags.get_tags(album.slug, "a_1.png") == {"person": ["Alex"]}


def test_assign_back_to_unsorted_carries_tags(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Keepers")
    store.assign(["a.png"], None, album.slug)
    store.tags.create_category("person")
    store.tags.set_tags(album.slug, "a.png", "person", ["Alex"])

    store.assign(["a.png"], album.slug, None)

    assert store.tags.get_tags(album.slug, "a.png") == {}
    assert store.tags.get_tags(None, "a.png") == {"person": ["Alex"]}


def test_delete_images_forgets_tags(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    store.tags.create_category("person")
    store.tags.set_tags(None, "a.png", "person", ["Alex"])

    store.delete_images(["a.png"], None)

    assert store.tags.get_tags(None, "a.png") == {}
    assert store.tags.all_tagged_images() == {}


def test_delete_album_moves_images_and_carries_tags(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Temp")
    store.assign(["a.png"], None, album.slug)
    store.tags.create_category("person")
    store.tags.set_tags(album.slug, "a.png", "person", ["Alex"])

    store.delete_album(album.slug)  # delete_images=False: moves back to unsorted

    assert store.tags.get_tags(album.slug, "a.png") == {}
    assert store.tags.get_tags(None, "a.png") == {"person": ["Alex"]}


def test_delete_album_with_files_forgets_tags(store: AlbumStore) -> None:
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Temp")
    store.assign(["a.png"], None, album.slug)
    store.tags.create_category("person")
    store.tags.set_tags(album.slug, "a.png", "person", ["Alex"])

    store.delete_album(album.slug, delete_images=True)

    assert store.tags.all_tagged_images() == {}


def test_untagged_images_unaffected_by_moves(store: AlbumStore) -> None:
    """Moving a never-tagged image must not create a spurious tags.json entry."""
    make_image(store.unsorted_dir / "a.png")
    album = store.create_album("Keepers")

    store.assign(["a.png"], None, album.slug)

    assert store.tags.all_tagged_images() == {}


def test_tags_persist_across_store_reopen(tmp_path: Path) -> None:
    store = AlbumStore(tmp_path / "outputs")
    make_image(store.unsorted_dir / "a.png")
    store.tags.create_category("pose")
    store.tags.set_tags(None, "a.png", "pose", ["Sitting"])

    reopened = AlbumStore(tmp_path / "outputs")

    assert reopened.tags.get_tags(None, "a.png") == {"pose": ["Sitting"]}
