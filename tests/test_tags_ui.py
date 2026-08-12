"""Tags tab UI handler tests."""

from __future__ import annotations

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from imagebatch.config import Config  # noqa: E402
from imagebatch.ui.context import AppContext  # noqa: E402
from imagebatch.ui.tags_tab import NO_GROUPING, build_tags_tab  # noqa: E402
from tests.conftest import make_image  # noqa: E402


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    return AppContext.create(Config(model_id="mock",
                                    output_dir=str(tmp_path / "outputs")))


@pytest.fixture
def tags(ctx: AppContext) -> dict:
    with gr.Blocks():
        return build_tags_tab(ctx)["handlers"]


def seed(ctx: AppContext, name: str, **tag_kwargs) -> str:
    make_image(ctx.store.unsorted_dir / name)
    for category, values in tag_kwargs.items():
        try:
            ctx.store.tags.create_category(category)
        except Exception:
            pass
        ctx.store.tags.set_tags(None, name, category, values)
    return name


def status_of(result) -> str:
    return result[-1]["value"]


# -- categories -------------------------------------------------------------
def test_create_category(ctx: AppContext, tags: dict) -> None:
    result = tags["create_category"]("Pose")
    assert "Created category **Pose**" in status_of(result)
    assert [c["name"] for c in ctx.store.tags.list_categories()] == ["Pose"]


def test_create_empty_category_warns(ctx: AppContext, tags: dict) -> None:
    assert "Enter a category name" in status_of(tags["create_category"]("  "))


def test_create_duplicate_category_errors(ctx: AppContext, tags: dict) -> None:
    tags["create_category"]("Pose")
    assert "already exists" in status_of(tags["create_category"]("pose"))


def test_delete_category(ctx: AppContext, tags: dict) -> None:
    seed(ctx, "a.png", pose=["Sitting"])
    result = tags["delete_category"]("pose")
    assert "Deleted" in status_of(result)
    assert ctx.store.tags.list_categories() == []


def test_delete_without_selection_warns(ctx: AppContext, tags: dict) -> None:
    assert "Pick a category" in status_of(tags["delete_category"](None))


# -- search -----------------------------------------------------------------
def test_search_flat(ctx: AppContext, tags: dict) -> None:
    seed(ctx, "a.png", pose=["Sitting"])
    seed(ctx, "b.png", pose=["Standing"])

    result = tags["search"]("pose", "Sitting", NO_GROUPING)

    assert "**1** image(s)" in result[0]["value"]


def test_search_grouped(ctx: AppContext, tags: dict) -> None:
    seed(ctx, "a.png", pose=["Sitting"], person=["Alex"])
    seed(ctx, "b.png", pose=["Sitting"], person=["Jordan"])

    result = tags["search"]("pose", "Sitting", "person")

    header = result[0]["value"]
    assert "**2** image(s)" in header
    assert "grouped by" in header
    # two group slots visible, each with one image
    visible = [u for u in result[1:] if u.get("visible")]
    assert len(visible) == 4  # 2 headings + 2 galleries


def test_search_reverse_direction(ctx: AppContext, tags: dict) -> None:
    """person=Alex grouped by pose — the 'vice versa' case."""
    seed(ctx, "a.png", pose=["Sitting"], person=["Alex"])
    seed(ctx, "b.png", pose=["Standing"], person=["Alex"])

    result = tags["search"]("person", "Alex", "pose")

    assert "**2** image(s)" in result[0]["value"]


def test_search_no_matches(ctx: AppContext, tags: dict) -> None:
    seed(ctx, "a.png", pose=["Sitting"])
    result = tags["search"]("pose", "Nonexistent", NO_GROUPING)
    assert "No images match" in result[0]["value"]


def test_search_requires_category_and_value(ctx: AppContext, tags: dict) -> None:
    result = tags["search"](None, None, NO_GROUPING)
    assert "Pick a category and value" in result[0]["value"]


# -- faces ------------------------------------------------------------------
def face_id_for(ctx: AppContext, seed_val: int) -> str:
    import random
    rng = random.Random(seed_val)
    return ctx.faces.match_or_create([rng.uniform(-1, 1) for _ in range(128)])


def test_rename_face(ctx: AppContext, tags: dict) -> None:
    fid = face_id_for(ctx, 1)
    result = tags["rename_face"](fid, "Alex")
    assert "**Alex**" in status_of(result)
    assert ctx.faces.get(fid).name == "Alex"


def test_rename_face_requires_selection(ctx: AppContext, tags: dict) -> None:
    assert "Pick a face" in status_of(tags["rename_face"](None, "Alex"))


def test_face_name_shown_instead_of_id(ctx: AppContext, tags: dict) -> None:
    fid = face_id_for(ctx, 1)
    ctx.faces.rename(fid, "Alex")
    assert tags["pretty_value"]("person", fid) == "Alex"


def test_unnamed_face_shows_id(ctx: AppContext, tags: dict) -> None:
    fid = face_id_for(ctx, 1)
    assert tags["pretty_value"]("person", fid) == fid


def test_merge_faces_retags_images(ctx: AppContext, tags: dict) -> None:
    keep = face_id_for(ctx, 1)
    absorb = face_id_for(ctx, 500)
    seed(ctx, "a.png", person=[absorb])

    result = tags["merge_faces"](keep, absorb)

    assert "re-tagged" in status_of(result)
    assert ctx.store.tags.get_tags(None, "a.png") == {"person": [keep]}


def test_merge_requires_both(ctx: AppContext, tags: dict) -> None:
    fid = face_id_for(ctx, 1)
    assert "Pick both faces" in status_of(tags["merge_faces"](fid, None))


def test_merge_self_errors(ctx: AppContext, tags: dict) -> None:
    fid = face_id_for(ctx, 1)
    assert "itself" in status_of(tags["merge_faces"](fid, fid))
