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
from imagebatch.ui.common import album_choices, page_count, page_slice  # noqa: E402
from imagebatch.ui.context import AppContext  # noqa: E402
from imagebatch.ui.gallery_tab import build_gallery_tab  # noqa: E402
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


def seed_unsorted(ctx: AppContext, count: int) -> list[str]:
    for i in range(count):
        make_image(ctx.store.unsorted_dir / f"img_{i:02d}.png")
    return ctx.store.list_unsorted()


# `render` returns: gallery, page_info, names, selected, page, selection,
#                   album_filter, target_album, status
def unpack(result):
    gallery_update, page_info, names, selected, page, selection, _, _, status = result
    return {
        "items": gallery_update["value"],
        "page_info": page_info["value"],
        "names": names,
        "selected": selected,
        "page": page,
        "selection": selection["value"],
        "status": status["value"],
    }


# -- pagination helpers ---------------------------------------------------
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


def test_album_choices_include_counts(ctx: AppContext) -> None:
    make_image(ctx.store.unsorted_dir / "a.png")
    ctx.store.create_album("Keepers")

    choices = album_choices(ctx.store)

    assert choices[0] == ("Unsorted (1)", UNSORTED)
    assert ("Keepers (0)", "keepers") in choices


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

    assert len(view["names"]) == 3
    assert len(view["items"]) == 3
    assert view["selected"] == []


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

    view = unpack(gallery["render"](UNSORTED, 1, ["img_00.png", "ghost.png"]))

    assert view["selected"] == ["img_00.png"]


def test_render_marks_selected_images(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["render"](UNSORTED, 1, ["img_00.png"]))
    captions = [caption for _, caption in view["items"]]

    assert captions == ["✅ img_00.png", "img_01.png"]


def test_thumbnails_are_cached(ctx: AppContext, gallery: dict) -> None:
    make_image(ctx.store.unsorted_dir / "a.png", size=(512, 512))

    view = unpack(gallery["render"](UNSORTED, 1, []))
    thumb = Path(view["items"][0][0])

    assert thumb.parent == ctx.thumbnails.cache_dir
    assert thumb.is_file()


def test_changing_album_clears_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")

    view = unpack(gallery["change_album"]("keepers"))

    assert view["selected"] == []
    assert view["names"] == []


# -- gallery: selection ---------------------------------------------------
def test_select_page_selects_visible_only(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 10)  # page size 4

    view = unpack(gallery["select_visible"](UNSORTED, 1, ctx.store.list_unsorted(), []))

    assert len(view["selected"]) == 4


def test_select_all(ctx: AppContext, gallery: dict) -> None:
    names = seed_unsorted(ctx, 10)

    view = unpack(gallery["select_all"](UNSORTED, 1, names))

    assert len(view["selected"]) == 10


def test_clear_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 3)

    view = unpack(gallery["clear_selection"](UNSORTED, 1))

    assert view["selected"] == []


# -- gallery: album assignment -------------------------------------------
def test_assign_moves_selection(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 3)
    ctx.store.create_album("Keepers")

    view = unpack(gallery["assign"](UNSORTED, "keepers", ["img_00.png", "img_01.png"], 1))

    assert "Moved **2**" in view["status"]
    assert view["selected"] == []
    assert sorted(ctx.store.list_images("keepers")) == ["img_00.png", "img_01.png"]
    assert ctx.store.list_unsorted() == ["img_02.png"]


def test_assign_without_selection_warns(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")

    view = unpack(gallery["assign"](UNSORTED, "keepers", [], 1))

    assert "Nothing selected" in view["status"]


def test_assign_without_target_warns(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)

    view = unpack(gallery["assign"](UNSORTED, None, ["img_00.png"], 1))

    assert "destination album" in view["status"]


def test_assign_to_same_album_warns(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)

    view = unpack(gallery["assign"](UNSORTED, UNSORTED, ["img_00.png"], 1))

    assert "already there" in view["status"]


def test_create_album_and_assign(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["create_and_assign"](UNSORTED, "My Picks", ["img_00.png"], 1))

    assert "Created **My Picks**" in view["status"]
    assert ctx.store.list_images("my-picks") == ["img_00.png"]
    assert ctx.store.list_unsorted() == ["img_01.png"]


def test_create_album_without_selection(ctx: AppContext, gallery: dict) -> None:
    view = unpack(gallery["create_and_assign"](UNSORTED, "Empty Album", [], 1))

    assert "nothing selected to move" in view["status"]
    assert [a.name for a in ctx.store.list_albums()] == ["Empty Album"]


def test_create_album_requires_name(ctx: AppContext, gallery: dict) -> None:
    view = unpack(gallery["create_and_assign"](UNSORTED, "  ", ["img_00.png"], 1))

    assert "Enter a name" in view["status"]
    assert ctx.store.list_albums() == []


def test_create_duplicate_album_reports_error(ctx: AppContext, gallery: dict) -> None:
    ctx.store.create_album("Keepers")

    view = unpack(gallery["create_and_assign"](UNSORTED, "keepers", [], 1))

    assert "already exists" in view["status"]


def test_move_back_to_unsorted(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    gallery["assign"](UNSORTED, "keepers", ["img_00.png"], 1)

    view = unpack(gallery["assign"]("keepers", UNSORTED, ["img_00.png"], 1))

    assert "Moved **1**" in view["status"]
    assert ctx.store.list_unsorted() == ["img_00.png"]


# -- gallery: deletion ----------------------------------------------------
def test_delete_requires_confirmation(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["delete_selection"](UNSORTED, ["img_00.png"], 1, False))

    assert "confirmation box" in view["status"]
    assert len(ctx.store.list_unsorted()) == 2


def test_delete_removes_files(ctx: AppContext, gallery: dict) -> None:
    seed_unsorted(ctx, 2)

    view = unpack(gallery["delete_selection"](UNSORTED, ["img_00.png"], 1, True))

    assert "Deleted **1**" in view["status"]
    assert ctx.store.list_unsorted() == ["img_01.png"]


# -- albums tab -----------------------------------------------------------
def test_album_covers_and_counts(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png", "img_01.png"], UNSORTED, "keepers")

    items, note = albums["covers"]()

    assert len(items) == 1
    assert "Keepers — 2 image(s)" in items[0][1]
    assert note == ""


def test_empty_albums_are_noted(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Empty")

    items, note = albums["covers"]()

    assert items == []
    assert "Empty" in note


def test_no_albums_note(ctx: AppContext, albums: dict) -> None:
    _, note = albums["covers"]()
    assert "No albums yet" in note


def test_rename_album(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Old Name")

    _, _, picker, _, status = albums["rename"]("old-name", "New Name")

    assert "Renamed to **New Name**" in status["value"]
    assert ctx.store.get_album("old-name").name == "New Name"


def test_rename_requires_name(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Keepers")

    *_, status = albums["rename"]("keepers", "   ")

    assert "Enter a new name" in status["value"]


def test_rename_rejects_unsorted(ctx: AppContext, albums: dict) -> None:
    *_, status = albums["rename"](UNSORTED, "Nope")
    assert "cannot be renamed" in status["value"]


def test_rename_duplicate_reports_error(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("A")
    ctx.store.create_album("B")

    *_, status = albums["rename"]("a", "B")

    assert "already exists" in status["value"]


def test_delete_album_requires_confirmation(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Keepers")

    *_, status = albums["delete"]("keepers", False, False)

    assert "to confirm" in status["value"]
    assert len(ctx.store.list_albums()) == 1


def test_delete_album_moves_images_to_unsorted(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png"], UNSORTED, "keepers")

    *_, status = albums["delete"]("keepers", True, False)

    assert "moved to Unsorted" in status["value"]
    assert ctx.store.list_unsorted() == ["img_00.png"]
    assert ctx.store.list_albums() == []


def test_delete_album_with_files(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png"], UNSORTED, "keepers")

    *_, status = albums["delete"]("keepers", True, True)

    assert "deleted" in status["value"]
    assert ctx.store.list_unsorted() == []


def test_delete_rejects_unsorted(ctx: AppContext, albums: dict) -> None:
    *_, status = albums["delete"](UNSORTED, True, True)
    assert "cannot be deleted" in status["value"]


def test_download_album_zip(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 2)
    ctx.store.create_album("Keepers")
    ctx.store.assign(["img_00.png", "img_01.png"], UNSORTED, "keepers")

    path, status = albums["download"]("keepers")

    assert "Ready" in status
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == ["img_00.png", "img_01.png"]


def test_download_unsorted_is_allowed(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 1)

    path, status = albums["download"](UNSORTED)

    assert "Ready" in status
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["img_00.png"]


def test_download_empty_album_reports_error(ctx: AppContext, albums: dict) -> None:
    ctx.store.create_album("Empty")

    path, status = albums["download"]("empty")

    assert path is None
    assert "no images" in status


def test_album_details_text(ctx: AppContext, albums: dict) -> None:
    seed_unsorted(ctx, 3)

    assert "3 image(s)" in albums["details"](UNSORTED)
    assert "not a real album" in albums["details"](UNSORTED)


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


def test_change_album_does_not_write_back_to_filter(ctx: AppContext, gallery: dict) -> None:
    """Handlers bound to album_filter.change must not update album_filter."""
    result = gallery["change_album"](UNSORTED)
    album_filter_update = result[6]

    assert album_filter_update == gr.update()  # a no-op update, so no event loop


def test_render_updates_filter_choices_normally(ctx: AppContext, gallery: dict) -> None:
    ctx.store.create_album("Keepers")
    result = gallery["render"](UNSORTED, 1, [])
    album_filter_update = result[6]

    assert "choices" in album_filter_update
    assert ("Keepers (0)", "keepers") in album_filter_update["choices"]
