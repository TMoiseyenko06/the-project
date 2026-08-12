from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagebatch.prompts import Preset, PromptError, find_preset, load_presets


def write(path: Path, data) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_presets(tmp_path / "absent.json") == []


def test_loads_basic_preset(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [
        {"name": "Watercolor", "prompt": "make it a watercolor painting",
         "tags": {"style": ["watercolor"]}},
    ])

    presets = load_presets(path)

    assert presets == [Preset(name="Watercolor", prompt="make it a watercolor painting",
                              tags={"style": ["watercolor"]})]


def test_preset_without_tags_is_fine(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [{"name": "Plain", "prompt": "sharpen it"}])
    presets = load_presets(path)
    assert presets == [Preset(name="Plain", prompt="sharpen it", tags={})]


def test_tag_value_as_single_string_becomes_list(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [
        {"name": "P", "prompt": "x", "tags": {"pose": "Standing"}},
    ])
    presets = load_presets(path)
    assert presets[0].tags == {"pose": ["Standing"]}


def test_multiple_categories_and_values(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [
        {"name": "P", "prompt": "x",
         "tags": {"pose": ["Standing"], "style": ["b&w", "grainy"]}},
    ])
    presets = load_presets(path)
    assert presets[0].tags == {"pose": ["Standing"], "style": ["b&w", "grainy"]}


def test_wrapped_in_presets_object(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", {"presets": [{"name": "A", "prompt": "x"}]})
    presets = load_presets(path)
    assert len(presets) == 1
    assert presets[0].name == "A"


def test_multiple_presets_preserve_order(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [
        {"name": "First", "prompt": "a"},
        {"name": "Second", "prompt": "b"},
    ])
    presets = load_presets(path)
    assert [p.name for p in presets] == ["First", "Second"]


def test_missing_name_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [{"prompt": "x"}])
    with pytest.raises(PromptError, match="name"):
        load_presets(path)


def test_missing_prompt_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [{"name": "X"}])
    with pytest.raises(PromptError, match="prompt"):
        load_presets(path)


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [
        {"name": "Dup", "prompt": "a"}, {"name": "dup", "prompt": "b"},
    ])
    with pytest.raises(PromptError, match="duplicate"):
        load_presets(path)


def test_non_list_top_level_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", "just a string")
    with pytest.raises(PromptError, match="list"):
        load_presets(path)


def test_presets_key_not_a_list_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", {"presets": "oops"})
    with pytest.raises(PromptError, match="list"):
        load_presets(path)


def test_dict_without_presets_key_is_empty(tmp_path: Path) -> None:
    """A dict with no "presets" key just means no presets, not an error."""
    path = write(tmp_path / "prompts.json", {"name": "X", "prompt": "y"})
    assert load_presets(path) == []


def test_invalid_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text("{ not json")
    with pytest.raises(PromptError, match="JSON"):
        load_presets(path)


def test_non_dict_entry_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", ["just a string"])
    with pytest.raises(PromptError, match="object"):
        load_presets(path)


def test_bad_tags_shape_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json", [{"name": "X", "prompt": "y", "tags": "oops"}])
    with pytest.raises(PromptError, match="tags"):
        load_presets(path)


def test_bad_tag_values_shape_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "prompts.json",
                [{"name": "X", "prompt": "y", "tags": {"pose": [1, 2]}}])
    with pytest.raises(PromptError, match="pose"):
        load_presets(path)


def test_find_preset_case_insensitive(tmp_path: Path) -> None:
    presets = [Preset(name="Watercolor", prompt="p")]
    assert find_preset(presets, "WATERCOLOR") is not None
    assert find_preset(presets, "watercolor ") is not None
    assert find_preset(presets, "nope") is None
