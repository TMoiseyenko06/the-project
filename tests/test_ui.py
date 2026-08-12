"""UI-layer tests.

The tab builders are run inside a Gradio Blocks context and their handler
closures are then called directly, so selection and album logic is covered
without needing a browser.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from imagebatch.config import Config  # noqa: E402
from imagebatch.storage import UNSORTED  # noqa: E402
from imagebatch.ui.albums_tab import build_albums_tab  # noqa: E402
from imagebatch.ui.app import build_ui  # noqa: E402
from imagebatch.ui.common import (INDENT, TOP_LEVEL, album_choices,  # noqa: E402
                                  page_count, page_slice, parent_value)
from imagebatch.ui.context import AppContext  # noqa: E402
from imagebatch.ui.gallery_tab import build_gallery_tab, split_token  # noqa: E402
from imagebatch.ui.run_tab import build_run_tab  # noqa: E402
from tests.conftest import make_image  # noqa: E402


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    config = Config(model_id="mock", output_dir=str(tmp_path / "outputs"),
                    gallery_page_size=4)
    return AppContext.create(config)


@pytest.fixture
def gallery(ctx: AppContext) -> dict:
    with gr.Blocks():
        return build_gallery_tab(ctx)["handlers"]


@pytest.fixture
def albums(ctx: AppContext) -> dict:
    with gr.Blocks():
        return build_albums_tab(ctx)["handlers"]


@pytest.fixture
def run_tab(ctx: AppContext) -> dict:
    with gr.Blocks():
        return build_run_tab(ctx)["handlers"]


def seed_unsorted(ctx: AppContext, count: int) -> list[str]:
    for i in range(count):
        make_image(ctx.store.unsorted_dir / f"img_{i:02d}.png")
    return ctx.store.list_unsorted()


def tok(album: str, name: str) -> str:
    return f"{album}/{name}"


# gallery `render` returns: gallery, page_info, tokens, selected, page, selection,
#                           album_filter, target_album, new_album_parent, status
def unpack(result):
    (gallery_update, page_info, tokens, selected, page, selection,
     album_filter, _, _, status) = result
    return {
        "items": gallery_update["value"],
        "page_info": page_info["value"],
        "tokens": tokens,
        "selected": selected,
        "page": page,
        "selection": selection["value"],
        "album_filter": album_filter,
        "status": status["value"],
    }


# albums `render` returns: overview, tree, picker, details, new_parent, move_parent, status
def unpack_albums(result):
    overview, tree, picker, details, new_parent, move_parent, status = result
    return {
        "items": overview["value"],
        "tree": tree["value"],
        "picker": picker,
        "details": details["value"],
        "move_choices": [value for _, value in move_parent["choices"]],
        "status": status["value"],
    }


# -- helpers --------------------------------------------------------------
@pytest.mark.parametrize(
    ("total", "size", "expected"),
    [(0, 10, 1), (1, 10, 1), (10, 10, 1), (11, 10, 2), (25, 10, 3)],
)
def test_page_count(total: int, size: int, expected: int) -> None:
    assert page_count(total, size) == expected


def test_page_slice_clamps_out_of_range() -> None:
    names = [f"{i}.png" for i in range(10)]
    assert page_slice(names, 1, 4) == names[:4]
    assert page_slice(names, 3, 4) == names[8:]
    assert page_slice(names, 99, 4) == names[8:]  # clamped to the last page
    assert page_slice(names, 0, 4) == names[:4]   # clamped to the first page


def test_split_token() -> None:
    assert split_token("keepers/img.png") == ("keepers", "img.png")
    assert split_token("unsorted/a b.png") == ("unsorted", "a b.png")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(TOP_LEVEL, None), (UNSORTED, None), (None, None), ("", None), ("keepers", "keepers")],
)
def test_parent_value(value, expected) -> None:
    assert parent_value(value) == expected


def test_album_choices_are_tree_ordered_and_indented(ctx: AppContext) -> None:
    parent = ctx.store.create_album("Keepers")
    ctx.store.create_album("Portraits", parent=parent.slug)
    ctx.store.create_album("Aardvark")

    labels = [label for label, _ in album_choices(ctx.store)]

    assert labels[0].startswith("Unsorted")
    assert labels[1:] == ["Aardvark (0)", "Keepers (0)", f"{INDENT}Portraits (0)"]


def test_album_choices_show_nested_counts(ctx: AppContext) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 1)
    ctx.store.assign(["img_00.png"], UNSORTED, child.slug)

    labels = dict((value, label) for label, value in album_choices(ctx.store))

    assert labels[parent.slug].strip() == "Keepers (0 / 1)"
    assert labels[child.slug].strip() == "Portraits (1)"


def test_album_choices_can_exclude(ctx: AppContext) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)

    values = [value for _, value in
              album_choices(ctx.store, include_unsorted=False, exclude={child.slug})]

    assert values == [parent.slug]


# -- whole app ------------------------------------------------------------
def test_build_ui_succeeds(tmp_path: Path) -> None:
    config = Config(model_id="mock", output_dir=str(tmp_path / "outputs"))
    demo, ctx = build_ui(config)

    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Batch Image Editor"
    assert len(demo.fns) > 0  # event handlers registered
    assert ctx.store.root.is_dir()


# -- gallery: rendering ---------------------------------------------------
def test_render_lists_unsorted(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 3)

    view = unpack(gallery["render"](UNSORTED, 1, []))

    assert len(view["tokens"]) == 3
    assert len(view["items"]) == 3
    assert view["tokens"][0] == tok(UNSORTED, "img_00.png")


def test_render_paginates(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 10)  # page size is 4

    view = unpack(gallery["render"](UNSORTED, 2, []))

    assert len(view["items"]) == 4
    assert "Page **2** / 3" in view["page_info"]


def test_render_clamps_page(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 6)

    view = unpack(gallery["render"](UNSORTED, 99, []))

    assert view["page"] == 2
    assert len(view["items"]) == 2


def test_render_drops_stale_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)

    view = unpack(gallery["render"](UNSORTED, 1,
                                    [tok(UNSORTED, "img_00.png"), tok(UNSORTED, "ghost.png")]))

    assert view["selected"] == [tok(UNSORTED, "img_00.png")]


def test_render_marks_selected_images(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["render"](UNSORTED, 1, [tok(UNSORTED, "img_00.png")]))
    captions = [caption for _, caption in view["items"]]

    assert captions == ["✅ img_00.png", "img_01.png"]


def test_thumbnails_are_cached(ctx: AppContext, gallery: dict) -> None:
    make_image(ctx.store.unsorted_dir / "a.png", size=(512, 512))

    view = unpack(gallery["render"](UNSORTED, 1, []))
    thumb = Path(view["items"][0][0])

    assert thumb.parent == ctx.thumbnails.cache_dir
    assert thumb.is_file()


def test_render_falls_back_when_album_vanishes(ctx: AppContext, gallery: dict) -> None:
    """The album may have been deleted from the Albums tab."""
    seed_unsorted(ctx, 1)

    view = unpack(gallery["render"]("ghost-album", 1, []))

    assert view["tokens"] == [tok(UNSORTED, "img_00.png")]


def test_changing_album_clears_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")

    view = unpack(gallery["change_album"]("keepers", False))

    assert view["selected"] == []
    assert view["tokens"] == []


def test_change_album_does_not_write_back_to_filter(ctx: AppContext, gallery: dict) -> None:
    """Handlers bound to album_filter.change must not update album_filter."""
    view = unpack(gallery["change_album"](UNSORTED, False))

    assert view["album_filter"] == gr.update()  # a no-op update, so no event loop


# -- gallery: nested viewing ---------------------------------------------
def test_include_sub_albums_shows_nested_images(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 2)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)
    ctx.store.assign(["img_01.png"], UNSORTED, child.slug)

    flat = unpack(gallery["render"](parent.slug, 1, [], nested=False))
    nested = unpack(gallery["render"](parent.slug, 1, [], nested=True))

    assert len(flat["tokens"]) == 1
    assert sorted(nested["tokens"]) == sorted([tok(child.slug, "img_01.png"),
                                               tok(parent.slug, "img_00.png")])


def test_nested_view_labels_images_by_album(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 1)
    ctx.store.assign(["img_00.png"], UNSORTED, child.slug)

    view = unpack(gallery["render"](parent.slug, 1, [], nested=True))

    assert view["items"][0][1] == "Portraits / img_00.png"


def test_toggle_nested_resets_selection(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    seed_unsorted(ctx, 1)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)

    view = unpack(gallery["toggle_nested"](parent.slug, True))

    assert view["selected"] == []


def test_same_filename_across_albums_stays_distinct(ctx: AppContext, gallery: dict) -> None:
    """Two sub-albums can hold the same filename; tokens keep them apart."""
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    make_image(ctx.store.unsorted_dir / "same.png")
    ctx.store.assign(["same.png"], UNSORTED, parent.slug)
    make_image(ctx.store.unsorted_dir / "same.png")
    ctx.store.assign(["same.png"], UNSORTED, child.slug)

    view = unpack(gallery["render"](parent.slug, 1, [], nested=True))

    assert len(view["tokens"]) == 2
    assert len(set(view["tokens"])) == 2


def test_move_selection_spanning_sub_albums(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    destination = ctx.store.create_album("Final")
    seed_unsorted(ctx, 2)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)
    ctx.store.assign(["img_01.png"], UNSORTED, child.slug)
    selection = [tok(parent.slug, "img_00.png"), tok(child.slug, "img_01.png")]

    view = unpack(gallery["assign"](parent.slug, destination.slug, selection, 1, True))

    assert "Moved **2**" in view["status"]
    assert sorted(ctx.store.list_images(destination.slug)) == ["img_00.png", "img_01.png"]
    assert ctx.store.list_images(parent.slug) == []
    assert ctx.store.list_images(child.slug) == []


def test_delete_selection_spanning_sub_albums(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 2)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)
    ctx.store.assign(["img_01.png"], UNSORTED, child.slug)
    selection = [tok(parent.slug, "img_00.png"), tok(child.slug, "img_01.png")]

    view = unpack(gallery["delete_selection"](parent.slug, selection, 1, True, True))

    assert "Deleted **2**" in view["status"]
    assert ctx.store.count_images(parent.slug, include_descendants=True) == 0


# -- gallery: selection ---------------------------------------------------
def test_select_page_selects_visible_only(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 10)  # page size 4
    tokens = gallery["view_tokens"](UNSORTED, False)

    view = unpack(gallery["select_visible"](UNSORTED, 1, tokens, [], False))

    assert len(view["selected"]) == 4


def test_select_all(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 10)
    tokens = gallery["view_tokens"](UNSORTED, False)

    view = unpack(gallery["select_all"](UNSORTED, 1, tokens, False))

    assert len(view["selected"]) == 10


def test_clear_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 3)

    view = unpack(gallery["clear_selection"](UNSORTED, 1, False))

    assert view["selected"] == []


# -- gallery: album assignment -------------------------------------------
def test_assign_moves_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 3)
    ctx.store.create_album("Keepers")
    selection = [tok(UNSORTED, "img_00.png"), tok(UNSORTED, "img_01.png")]

    view = unpack(gallery["assign"](UNSORTED, "keepers", selection, 1, False))

    assert "Moved **2**" in view["status"]
    assert view["selected"] == []
    assert sorted(ctx.store.list_images("keepers")) == ["img_00.png", "img_01.png"]
    assert ctx.store.list_unsorted() == ["img_02.png"]


def test_assign_without_selection_warns(ctx: AppContext, gallery: dict) -> None:
    ctx.store.create_album("Keepers")

    view = unpack(gallery["assign"](UNSORTED, "keepers", [], 1, False))

    assert "Nothing selected" in view["status"]


def test_assign_without_target_warns(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)

    view = unpack(gallery["assign"](UNSORTED, None, [tok(UNSORTED, "img_00.png")], 1, False))

    assert "destination album" in view["status"]


def test_assign_to_same_album_warns(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)

    view = unpack(gallery["assign"](UNSORTED, UNSORTED, [tok(UNSORTED, "img_00.png")],
                                    1, False))

    assert "already there" in view["status"]


def test_create_album_and_assign(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["create_and_assign"](UNSORTED, "My Picks", TOP_LEVEL,
                                               [tok(UNSORTED, "img_00.png")], 1, False))

    assert "Created **My Picks**" in view["status"]
    assert ctx.store.list_images("my-picks") == ["img_00.png"]
    assert ctx.store.list_unsorted() == ["img_01.png"]


def test_create_nested_album_and_assign(ctx: AppContext, gallery: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    seed_unsorted(ctx, 1)

    view = unpack(gallery["create_and_assign"](UNSORTED, "Portraits", parent.slug,
                                               [tok(UNSORTED, "img_00.png")], 1, False))

    assert "Keepers / Portraits" in view["status"]
    created = ctx.store.find_by_name("Portraits", parent=parent.slug)
    assert created is not None
    assert ctx.store.list_images(created.slug) == ["img_00.png"]


def test_create_album_without_selection(ctx: AppContext, gallery: dict) -> None:
    view = unpack(gallery["create_and_assign"](UNSORTED, "Empty Album", TOP_LEVEL,
                                               [], 1, False))

    assert "nothing selected to move" in view["status"]
    assert [a.name for a in ctx.store.list_albums()] == ["Empty Album"]


def test_create_album_requires_name(ctx: AppContext, gallery: dict) -> None:
    view = unpack(gallery["create_and_assign"](UNSORTED, "  ", TOP_LEVEL, [], 1, False))

    assert "Enter a name" in view["status"]
    assert ctx.store.list_albums() == []


def test_create_duplicate_album_reports_error(ctx: AppContext, gallery: dict) -> None:
    ctx.store.create_album("Keepers")

    view = unpack(gallery["create_and_assign"](UNSORTED, "keepers", TOP_LEVEL, [], 1, False))

    assert "already exists" in view["status"]


def test_move_back_to_unsorted(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png"], UNSORTED, "keepers")

    view = unpack(gallery["assign"]("keepers", UNSORTED, [tok("keepers", "img_00.png")],
                                    1, False))

    assert "Moved **1**" in view["status"]
    assert ctx.store.list_unsorted() == ["img_00.png"]


# -- gallery: deletion ----------------------------------------------------
def test_delete_requires_confirmation(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["delete_selection"](UNSORTED, [tok(UNSORTED, "img_00.png")],
                                              1, False, False))

    assert "confirmation box" in view["status"]
    assert len(ctx.store.list_unsorted()) == 2


def test_delete_removes_files(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["delete_selection"](UNSORTED, [tok(UNSORTED, "img_00.png")],
                                              1, True, False))

    assert "Deleted **1**" in view["status"]
    assert ctx.store.list_unsorted() == ["img_01.png"]


# -- albums tab -----------------------------------------------------------
def test_album_covers_and_counts(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png", "img_01.png"], UNSORTED, "keepers")

    items, tree = albums["covers"]()

    assert len(items) == 1
    assert "Keepers — 2" in items[0][1]
    assert "Keepers" in tree


def test_album_tree_shows_hierarchy(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    ctx.store.create_album("Portraits", parent=parent.slug)

    tree = albums["tree"]()

    assert "- **Keepers**" in tree
    assert "    - **Portraits**" in tree


def test_parent_cover_borrows_from_child(ctx: AppContext, albums: dict) -> None:
    """An album with no images of its own still gets a thumbnail."""
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 1)
    ctx.store.assign(["img_00.png"], UNSORTED, child.slug)

    items, _ = albums["covers"]()
    captions = [caption for _, caption in items]

    assert any("Keepers — 0 (1 nested)" in c for c in captions)


def test_no_albums_note(ctx: AppContext, albums: dict) -> None:
    _, tree = albums["covers"]()
    assert "No albums yet" in tree


def test_create_album_from_albums_tab(ctx: AppContext, albums: dict) -> None:
    view = unpack_albums(albums["create"]("Keepers", TOP_LEVEL))

    assert "Created **Keepers**" in view["status"]
    assert [a.name for a in ctx.store.list_albums()] == ["Keepers"]


def test_create_nested_album_from_albums_tab(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")

    view = unpack_albums(albums["create"]("Portraits", parent.slug))

    assert "Keepers / Portraits" in view["status"]


def test_rename_album(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Old Name")

    view = unpack_albums(albums["rename"]("old-name", "New Name"))

    assert "Renamed to **New Name**" in view["status"]
    assert ctx.store.get_album("old-name").name == "New Name"


def test_rename_rejects_unsorted(ctx: AppContext, albums: dict) -> None:
    view = unpack_albums(albums["rename"](UNSORTED, "Nope"))
    assert "cannot be renamed" in view["status"]


def test_move_album_from_albums_tab(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    loose = ctx.store.create_album("Portraits")

    view = unpack_albums(albums["move"](loose.slug, parent.slug))

    assert "Moved to **Keepers / Portraits**" in view["status"]
    assert ctx.store.get_album(loose.slug).parent == parent.slug


def test_move_album_to_top_level(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)

    albums["move"](child.slug, TOP_LEVEL)

    assert ctx.store.get_album(child.slug).parent is None


def test_move_into_descendant_reports_error(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)

    view = unpack_albums(albums["move"](parent.slug, child.slug))

    assert "own sub-album" in view["status"]
    assert ctx.store.get_album(parent.slug).parent is None


def test_move_choices_exclude_own_subtree(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    other = ctx.store.create_album("Rejects")

    from imagebatch.ui.albums_tab import build_albums_tab as build

    with gr.Blocks():
        render = build(ctx)["render"]
    view = unpack_albums(render(parent.slug))

    assert parent.slug not in view["move_choices"]
    assert child.slug not in view["move_choices"]
    assert other.slug in view["move_choices"]


def test_move_rejects_unsorted(ctx: AppContext, albums: dict) -> None:
    view = unpack_albums(albums["move"](UNSORTED, TOP_LEVEL))
    assert "cannot be moved" in view["status"]


def test_delete_album_requires_confirmation(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Keepers")

    view = unpack_albums(albums["delete"]("keepers", False, False, False))

    assert "to confirm" in view["status"]
    assert len(ctx.store.list_albums()) == 1


def test_delete_album_moves_images_to_unsorted(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png"], UNSORTED, "keepers")

    view = unpack_albums(albums["delete"]("keepers", True, False, False))

    assert "moved to Unsorted" in view["status"]
    assert ctx.store.list_unsorted() == ["img_00.png"]
    assert ctx.store.list_albums() == []


def test_delete_album_promotes_sub_albums(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)

    view = unpack_albums(albums["delete"](parent.slug, True, False, False))

    assert "promoted" in view["status"]
    assert ctx.store.get_album(child.slug).parent is None


def test_delete_album_recursive(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    ctx.store.create_album("Portraits", parent=parent.slug)

    view = unpack_albums(albums["delete"](parent.slug, True, True, False))

    assert "1 sub-album(s) deleted" in view["status"]
    assert ctx.store.list_albums() == []


def test_delete_album_with_files(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png"], UNSORTED, "keepers")

    view = unpack_albums(albums["delete"]("keepers", True, False, True))

    assert "deleted" in view["status"]
    assert ctx.store.list_unsorted() == []


def test_delete_rejects_unsorted(ctx: AppContext, albums: dict) -> None:
    view = unpack_albums(albums["delete"](UNSORTED, True, True, True))
    assert "cannot be deleted" in view["status"]


def test_download_album_zip(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png", "img_01.png"], UNSORTED, "keepers")

    path, status = albums["download"]("keepers", True)

    assert "Ready" in status
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == ["img_00.png", "img_01.png"]


def test_download_album_zip_nests_children(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 2)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)
    ctx.store.assign(["img_01.png"], UNSORTED, child.slug)

    path, _ = albums["download"](parent.slug, True)

    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == ["Portraits/img_01.png", "img_00.png"]


def test_download_album_zip_without_children(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 2)
    ctx.store.assign(["img_00.png"], UNSORTED, parent.slug)
    ctx.store.assign(["img_01.png"], UNSORTED, child.slug)

    path, _ = albums["download"](parent.slug, False)

    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["img_00.png"]


def test_download_unsorted_is_allowed(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)

    path, status = albums["download"](UNSORTED, True)

    assert "Ready" in status
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["img_00.png"]


def test_download_empty_album_reports_error(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Empty")

    path, status = albums["download"]("empty", True)

    assert path is None
    assert "no images" in status


def test_album_details_text(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 3)

    assert "3 image(s)" in albums["details"](UNSORTED)
    assert "not a real album" in albums["details"](UNSORTED)


def test_album_details_reports_nesting(ctx: AppContext, albums: dict) -> None:
    parent = ctx.store.create_album("Keepers")
    child = ctx.store.create_album("Portraits", parent=parent.slug)
    seed_unsorted(ctx, 1)
    ctx.store.assign(["img_00.png"], UNSORTED, child.slug)

    text = albums["details"](parent.slug)

    assert "0 image(s) directly, 1 including sub-albums" in text
    assert "1 sub-album(s)" in text


# -- run tab --------------------------------------------------------------
def test_run_tab_refresh_lists_albums(ctx: AppContext, run_tab: dict) -> None:
    ctx.store.create_album("Keepers")

    target, parent, name = run_tab["refresh_albums"](UNSORTED)

    assert "keepers" in [value for _, value in target["choices"]]
    assert target["value"] == UNSORTED
    assert name["value"] == ""


def test_run_tab_refresh_keeps_valid_selection(ctx: AppContext, run_tab: dict) -> None:
    album = ctx.store.create_album("Keepers")

    target, _, _ = run_tab["refresh_albums"](album.slug)

    assert target["value"] == album.slug


def test_run_tab_refresh_resets_deleted_selection(ctx: AppContext, run_tab: dict) -> None:
    target, _, _ = run_tab["refresh_albums"]("ghost")

    assert target["value"] == UNSORTED


def test_run_batch_creates_named_album(ctx: AppContext, run_tab: dict,
                                       tmp_path: Path) -> None:
    sources = tmp_path / "src"
    for i in range(2):
        make_image(sources / f"photo_{i}.png")

    updates = list(run_tab["run_batch"]("Folder path", str(sources), None, None, "a prompt",
                                        UNSORTED, "Batch One", TOP_LEVEL, True, True, "— none —", "", False))

    assert "filed into **Batch One**" in updates[-1][1]
    created = ctx.store.find_by_name("Batch One")
    assert created is not None
    assert len(ctx.store.list_images(created.slug)) == 2
    assert ctx.store.list_unsorted() == []


def test_run_batch_uses_dropdown_album(ctx: AppContext, run_tab: dict,
                                       tmp_path: Path) -> None:
    album = ctx.store.create_album("Existing")
    sources = tmp_path / "src"
    make_image(sources / "photo.png")

    updates = list(run_tab["run_batch"]("Folder path", str(sources), None, None, "a prompt",
                                        album.slug, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "filed into **Existing**" in updates[-1][1]
    assert ctx.store.list_images(album.slug) == ["photo.png"]


def test_run_batch_defaults_to_unsorted(ctx: AppContext, run_tab: dict,
                                        tmp_path: Path) -> None:
    sources = tmp_path / "src"
    make_image(sources / "photo.png")

    list(run_tab["run_batch"]("Folder path", str(sources), None, None, "a prompt",
                              UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert ctx.store.list_unsorted() == ["photo.png"]


def test_run_batch_duplicate_album_name_reports_error(ctx: AppContext, run_tab: dict,
                                                      tmp_path: Path) -> None:
    ctx.store.create_album("Taken")
    sources = tmp_path / "src"
    make_image(sources / "photo.png")

    updates = list(run_tab["run_batch"]("Folder path", str(sources), None, None, "a prompt",
                                        UNSORTED, "Taken", TOP_LEVEL, True, True, "— none —", "", False))

    assert "already exists" in updates[-1][0]
    assert ctx.store.list_unsorted() == []  # nothing ran


def test_run_batch_requires_prompt(ctx: AppContext, run_tab: dict, tmp_path: Path) -> None:
    sources = tmp_path / "src"
    make_image(sources / "photo.png")

    updates = list(run_tab["run_batch"]("Folder path", str(sources), None, None, "  ",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Enter a prompt" in updates[-1][0]


def test_run_batch_requires_source(ctx: AppContext, run_tab: dict) -> None:
    updates = list(run_tab["run_batch"]("Folder path", "", None, None, "a prompt",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Choose a folder" in updates[-1][0]


# -- run tab: individual image uploads ------------------------------------
def test_run_batch_from_uploaded_images(ctx: AppContext, run_tab: dict,
                                        tmp_path: Path) -> None:
    uploads = tmp_path / "phone_uploads"
    paths = [str(make_image(uploads / f"IMG_{i}.jpg")) for i in range(3)]

    updates = list(run_tab["run_batch"]("Upload images", "", None, paths, "a prompt",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Processed: **3**" in updates[-1][1]
    assert len(ctx.store.list_unsorted()) == 3


def test_run_batch_upload_requires_files(ctx: AppContext, run_tab: dict) -> None:
    updates = list(run_tab["run_batch"]("Upload images", "", None, None, "a prompt",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Choose one or more images" in updates[-1][0]


def test_run_batch_upload_requires_files_empty_list(ctx: AppContext, run_tab: dict) -> None:
    updates = list(run_tab["run_batch"]("Upload images", "", None, [], "a prompt",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Choose one or more images" in updates[-1][0]


def test_run_batch_upload_into_album(ctx: AppContext, run_tab: dict, tmp_path: Path) -> None:
    uploads = tmp_path / "phone_uploads"
    paths = [str(make_image(uploads / "IMG_0.jpg"))]

    updates = list(run_tab["run_batch"]("Upload images", "", None, paths, "a prompt",
                                        UNSORTED, "From Phone", TOP_LEVEL, True, True, "— none —", "", False))

    assert "filed into **From Phone**" in updates[-1][1]
    created = ctx.store.find_by_name("From Phone")
    assert created is not None
    # output filenames always use config.output_format, not the source extension
    assert ctx.store.list_images(created.slug) == ["IMG_0.png"]


def test_run_batch_upload_ignores_non_images(ctx: AppContext, run_tab: dict,
                                             tmp_path: Path) -> None:
    uploads = tmp_path / "phone_uploads"
    uploads.mkdir(parents=True)
    good = make_image(uploads / "IMG_0.jpg")
    bad = uploads / "notes.txt"
    bad.write_text("not an image")

    updates = list(run_tab["run_batch"]("Upload images", "", None, [str(good), str(bad)],
                                        "a prompt", UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "Processed: **1**" in updates[-1][1]


def test_run_batch_upload_all_non_images_rejected(ctx: AppContext, run_tab: dict,
                                                  tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")

    updates = list(run_tab["run_batch"]("Upload images", "", None, [str(bad)], "a prompt",
                                        UNSORTED, "", TOP_LEVEL, True, True, "— none —", "", False))

    assert "no valid images" in updates[-1][0]


# -- context wiring -------------------------------------------------------
def test_context_syncs_albums_on_start(tmp_path: Path) -> None:
    """Files dropped into an album folder by hand are picked up at startup."""
    config = Config(model_id="mock", output_dir=str(tmp_path / "outputs"))
    first = AppContext.create(config)
    album = first.store.create_album("Keepers")
    make_image(first.store.album_dir(album.slug) / "manual.png")

    second = AppContext.create(config)

    assert second.store.get_album(album.slug).images == ["manual.png"]


def test_context_shares_manifest_with_runner(tmp_path: Path) -> None:
    config = Config(model_id="mock", output_dir=str(tmp_path / "outputs"))
    ctx = AppContext.create(config)
    assert ctx.runner.manifest is ctx.manifest


def test_thumbnail_falls_back_for_corrupt_image(ctx: AppContext) -> None:
    broken = ctx.store.unsorted_dir / "broken.png"
    broken.write_bytes(b"not an image")

    assert ctx.thumbnails.get(broken) == str(broken)


def test_thumbnail_reflects_changed_file(ctx: AppContext) -> None:
    path = ctx.store.unsorted_dir / "a.png"
    make_image(path, size=(256, 256), color="red")
    first = ctx.thumbnails.get(path)
    make_image(path, size=(256, 256), color="blue")

    assert ctx.thumbnails.get(path) != first  # cache key includes mtime/size
