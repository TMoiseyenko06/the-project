"""Nested album behaviour: hierarchy, moves, cycle safety, recursive delete."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from imagebatch.storage import MAX_DEPTH, UNSORTED, AlbumStore, StorageError
from tests.conftest import make_image


def tree_names(store: AlbumStore) -> list[str]:
    return [f"{'  ' * depth}{album.name}" for album, depth in store.album_tree()]


# -- creating nested albums ----------------------------------------------
def test_create_child_album(store: AlbumStore) -> None:
    parent = store.create_album("Keepers")
    child = store.create_album("Portraits", parent=parent.slug)

    assert child.parent == parent.slug
    assert [a.slug for a in store.children(parent.slug)] == [child.slug]
    assert store.children(child.slug) == []


def test_create_rejects_missing_parent(store: AlbumStore) -> None:
    with pytest.raises(StorageError, match="no such parent"):
        store.create_album("Orphan", parent="nope")


def test_unsorted_cannot_be_a_parent(store: AlbumStore) -> None:
    """Unsorted is not a real album, so nesting under it means top level."""
    album = store.create_album("Keepers", parent=UNSORTED)
    assert album.parent is None


def test_same_name_allowed_under_different_parents(store: AlbumStore) -> None:
    a = store.create_album("Shoot A")
    b = store.create_album("Shoot B")

    first = store.create_album("Selects", parent=a.slug)
    second = store.create_album("Selects", parent=b.slug)

    assert first.slug != second.slug
    assert first.name == second.name == "Selects"


def test_duplicate_name_rejected_among_siblings(store: AlbumStore) -> None:
    parent = store.create_album("Keepers")
    store.create_album("Portraits", parent=parent.slug)

    with pytest.raises(StorageError, match="already exists"):
        store.create_album("portraits", parent=parent.slug)


def test_top_level_duplicate_still_rejected(store: AlbumStore) -> None:
    store.create_album("Keepers")
    with pytest.raises(StorageError, match="already exists"):
        store.create_album("keepers")


def test_deep_nesting(store: AlbumStore) -> None:
    parent = None
    for i in range(5):
        parent = store.create_album(f"Level {i}", parent=parent).slug

    assert len(store.ancestors(parent)) == 4
    assert store.path_name(parent) == "Level 0 / Level 1 / Level 2 / Level 3 / Level 4"


def test_depth_limit_enforced(store: AlbumStore) -> None:
    parent = None
    for i in range(MAX_DEPTH):
        parent = store.create_album(f"Level {i}", parent=parent).slug

    with pytest.raises(StorageError, match="nested more than"):
        store.create_album("Too deep", parent=parent)


# -- tree ordering --------------------------------------------------------
def test_album_tree_is_depth_first_and_sorted(store: AlbumStore) -> None:
    b = store.create_album("Beta")
    a = store.create_album("Alpha")
    store.create_album("Zulu", parent=a.slug)
    store.create_album("Mike", parent=a.slug)
    store.create_album("Gamma", parent=b.slug)

    assert tree_names(store) == ["Alpha", "  Mike", "  Zulu", "Beta", "  Gamma"]


def test_album_tree_empty(store: AlbumStore) -> None:
    assert store.album_tree() == []


def test_path_name_for_unsorted(store: AlbumStore) -> None:
    assert store.path_name(UNSORTED) == "Unsorted"


# -- moving albums --------------------------------------------------------
def test_move_album_under_another(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta")

    moved = store.move_album(b.slug, a.slug)

    assert moved.parent == a.slug
    assert tree_names(store) == ["Alpha", "  Beta"]


def test_move_album_to_top_level(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)

    store.move_album(b.slug, None)

    assert store.get_album(b.slug).parent is None
    assert tree_names(store) == ["Alpha", "Beta"]


def test_move_keeps_images_and_subtree(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta")
    c = store.create_album("Gamma", parent=b.slug)
    make_image(store.unsorted_dir / "x.png")
    store.assign(["x.png"], None, c.slug)

    store.move_album(b.slug, a.slug)

    assert store.get_album(c.slug).parent == b.slug
    assert store.list_images(c.slug) == ["x.png"]
    assert store.path_name(c.slug) == "Alpha / Beta / Gamma"


def test_move_into_self_rejected(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    with pytest.raises(StorageError, match="inside itself"):
        store.move_album(a.slug, a.slug)


def test_move_into_own_descendant_rejected(store: AlbumStore) -> None:
    """The cycle case: moving a parent into its own child."""
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    c = store.create_album("Gamma", parent=b.slug)

    with pytest.raises(StorageError, match="own sub-album"):
        store.move_album(a.slug, c.slug)

    assert store.get_album(a.slug).parent is None  # unchanged


def test_move_rejects_sibling_name_clash(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    store.create_album("Selects", parent=a.slug)
    loose = store.create_album("Selects")

    with pytest.raises(StorageError, match="already exists"):
        store.move_album(loose.slug, a.slug)


def test_move_to_same_parent_is_a_noop(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)

    assert store.move_album(b.slug, a.slug).parent == a.slug


def test_move_respects_depth_limit(store: AlbumStore) -> None:
    chain = None
    for i in range(MAX_DEPTH - 1):
        chain = store.create_album(f"Level {i}", parent=chain).slug
    deep = store.create_album("Sub A")
    store.create_album("Sub B", parent=deep.slug)

    with pytest.raises(StorageError, match="more than"):
        store.move_album(deep.slug, chain)


# -- renaming -------------------------------------------------------------
def test_rename_allows_name_used_under_other_parent(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta")
    store.create_album("Selects", parent=a.slug)
    child = store.create_album("Temp", parent=b.slug)

    renamed = store.rename_album(child.slug, "Selects")

    assert renamed.name == "Selects"


def test_rename_rejects_sibling_clash(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    store.create_album("One", parent=a.slug)
    two = store.create_album("Two", parent=a.slug)

    with pytest.raises(StorageError, match="already exists"):
        store.rename_album(two.slug, "One")


# -- deleting -------------------------------------------------------------
def test_delete_promotes_children(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    c = store.create_album("Gamma", parent=b.slug)

    result = store.delete_album(b.slug)

    assert result["albums"] == 1
    assert store.get_album(c.slug).parent == a.slug
    assert tree_names(store) == ["Alpha", "  Gamma"]


def test_delete_promotes_children_to_top_level(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)

    store.delete_album(a.slug)

    assert store.get_album(b.slug).parent is None


def test_delete_promotion_resolves_name_clash(store: AlbumStore) -> None:
    """A promoted child may collide with an existing name at the destination."""
    a = store.create_album("Alpha")
    store.create_album("Selects", parent=a.slug)
    doomed = store.create_album("Doomed", parent=a.slug)
    clashing = store.create_album("Selects", parent=doomed.slug)

    store.delete_album(doomed.slug)

    assert store.get_album(clashing.slug).name == "Selects (2)"
    assert store.get_album(clashing.slug).parent == a.slug


def test_delete_recursive_removes_subtree(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    c = store.create_album("Gamma", parent=b.slug)

    result = store.delete_album(a.slug, recursive=True)

    assert result["albums"] == 3
    assert store.list_albums() == []
    for slug in (a.slug, b.slug, c.slug):
        assert not store.album_dir(slug).exists()


def test_delete_recursive_moves_images_to_unsorted(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    make_image(store.unsorted_dir / "y.png")
    store.assign(["x.png"], None, a.slug)
    store.assign(["y.png"], None, b.slug)

    result = store.delete_album(a.slug, recursive=True)

    assert result["images"] == 2
    assert sorted(store.list_unsorted()) == ["x.png", "y.png"]


def test_delete_recursive_with_files(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    store.assign(["x.png"], None, b.slug)

    store.delete_album(a.slug, recursive=True, delete_images=True)

    assert store.list_unsorted() == []
    assert store.list_albums() == []


# -- images across the hierarchy -----------------------------------------
def test_list_images_excludes_descendants_by_default(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    make_image(store.unsorted_dir / "y.png")
    store.assign(["x.png"], None, a.slug)
    store.assign(["y.png"], None, b.slug)

    assert store.list_images(a.slug) == ["x.png"]
    assert sorted(store.list_images(a.slug, include_descendants=True)) == ["x.png", "y.png"]


def test_count_images_with_descendants(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    c = store.create_album("Gamma", parent=b.slug)
    for i, slug in enumerate([a.slug, b.slug, c.slug]):
        make_image(store.unsorted_dir / f"img_{i}.png")
        store.assign([f"img_{i}.png"], None, slug)

    assert store.count_images(a.slug) == 1
    assert store.count_images(a.slug, include_descendants=True) == 3


def test_list_images_located_reports_owner(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    make_image(store.unsorted_dir / "y.png")
    store.assign(["x.png"], None, a.slug)
    store.assign(["y.png"], None, b.slug)

    located = dict((name, slug) for slug, name in
                   store.list_images_located(a.slug, include_descendants=True))

    assert located == {"x.png": a.slug, "y.png": b.slug}


def test_same_filename_in_parent_and_child(store: AlbumStore) -> None:
    """Nested albums are separate directories, so names may repeat."""
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "same.png")
    store.assign(["same.png"], None, a.slug)
    make_image(store.unsorted_dir / "same.png")
    store.assign(["same.png"], None, b.slug)

    located = store.list_images_located(a.slug, include_descendants=True)

    assert len(located) == 2
    assert {slug for slug, _ in located} == {a.slug, b.slug}


# -- zip export -----------------------------------------------------------
def test_zip_nests_sub_albums(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    c = store.create_album("Gamma", parent=b.slug)
    for i, slug in enumerate([a.slug, b.slug, c.slug]):
        make_image(store.unsorted_dir / f"img_{i}.png")
        store.assign([f"img_{i}.png"], None, slug)

    archive = store.zip_album(a.slug)

    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == [
            "Beta/Gamma/img_2.png",
            "Beta/img_1.png",
            "img_0.png",
        ]


def test_zip_can_exclude_descendants(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    make_image(store.unsorted_dir / "y.png")
    store.assign(["x.png"], None, a.slug)
    store.assign(["y.png"], None, b.slug)

    archive = store.zip_album(a.slug, include_descendants=False)

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["x.png"]


def test_zip_parent_with_only_child_images(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.unsorted_dir / "y.png")
    store.assign(["y.png"], None, b.slug)

    archive = store.zip_album(a.slug)

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["Beta/y.png"]


def test_zip_sanitises_album_names(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Bad/Name: here", parent=a.slug)
    make_image(store.unsorted_dir / "x.png")
    store.assign(["x.png"], None, b.slug)

    archive = store.zip_album(a.slug)

    with zipfile.ZipFile(archive) as zf:
        name = zf.namelist()[0]
    assert name == "Bad_Name_ here/x.png"
    assert ".." not in name


# -- persistence and migration -------------------------------------------
def test_hierarchy_survives_reopen(tmp_path: Path) -> None:
    store = AlbumStore(tmp_path / "outputs")
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)

    reopened = AlbumStore(tmp_path / "outputs")

    assert reopened.get_album(b.slug).parent == a.slug
    assert [f"{'  ' * d}{al.name}" for al, d in reopened.album_tree()] == ["Alpha", "  Beta"]


def test_v1_file_migrates_to_top_level(tmp_path: Path) -> None:
    """An albums.json written before nesting existed must still load."""
    root = tmp_path / "outputs"
    root.mkdir(parents=True)
    (root / "albums.json").write_text(json.dumps({
        "version": 1,
        "albums": {
            "keepers": {"name": "Keepers", "created_at": 1.0, "images": ["a.png"]},
            "rejects": {"name": "Rejects", "created_at": 2.0, "images": []},
        },
    }))

    store = AlbumStore(root)

    assert all(album.parent is None for album in store.list_albums())
    assert store.get_album("keepers").images == ["a.png"]
    assert len(store.album_tree()) == 2


def test_missing_parent_is_repaired(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir(parents=True)
    (root / "albums.json").write_text(json.dumps({
        "version": 2,
        "albums": {
            "orphan": {"name": "Orphan", "created_at": 1.0, "images": [], "parent": "ghost"},
        },
    }))

    store = AlbumStore(root)

    assert store.get_album("orphan").parent is None


def test_cycle_in_file_is_broken(tmp_path: Path) -> None:
    """A hand-edited cycle must not hang the tree walk."""
    root = tmp_path / "outputs"
    root.mkdir(parents=True)
    (root / "albums.json").write_text(json.dumps({
        "version": 2,
        "albums": {
            "a": {"name": "A", "created_at": 1.0, "images": [], "parent": "b"},
            "b": {"name": "B", "created_at": 2.0, "images": [], "parent": "a"},
        },
    }))

    store = AlbumStore(root)

    assert len(store.album_tree()) == 2  # terminates, and nothing is lost


def test_sync_preserves_parents(store: AlbumStore) -> None:
    a = store.create_album("Alpha")
    b = store.create_album("Beta", parent=a.slug)
    make_image(store.album_dir(b.slug) / "manual.png")

    store.sync()

    assert store.get_album(b.slug).parent == a.slug
    assert store.list_images(b.slug) == ["manual.png"]
