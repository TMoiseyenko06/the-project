"""Reusable prompt presets.

``prompts.json`` at the project root holds a list of presets — a name, the
prompt text, and the tags that should be applied to every image a batch run
under that preset produces. Selecting one in the Run tab fills in the prompt
and queues its tags; editing the prompt afterward doesn't detach it — the
preset stays selected (and its tags still apply) until you pick a different
one or clear it.

Kept deliberately simple (a flat JSON list) since this is meant to be
hand-edited directly, same spirit as ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROMPTS_FILE = "prompts.json"


class PromptError(RuntimeError):
    """Raised for malformed prompts.json content."""


@dataclass
class Preset:
    name: str
    prompt: str
    # {category_key: [values]}, applied to every image this preset produces —
    # merged with any other tags at run time, not exclusive of them.
    tags: dict[str, list[str]] = field(default_factory=dict)


def _validate_entry(raw: Any, index: int) -> Preset:
    if not isinstance(raw, dict):
        raise PromptError(f"prompts.json entry {index} must be an object, got {raw!r}")
    name = str(raw.get("name") or "").strip()
    prompt = str(raw.get("prompt") or "").strip()
    if not name:
        raise PromptError(f"prompts.json entry {index} is missing a name")
    if not prompt:
        raise PromptError(f"prompts.json entry {index} ({name!r}) is missing a prompt")

    raw_tags = raw.get("tags") or {}
    if not isinstance(raw_tags, dict):
        raise PromptError(f"prompts.json entry {index} ({name!r}): `tags` must be an object "
                          "of category -> [values]")
    tags: dict[str, list[str]] = {}
    for category, values in raw_tags.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise PromptError(f"prompts.json entry {index} ({name!r}): tag values for "
                              f"{category!r} must be a string or list of strings")
        cleaned = sorted({v.strip() for v in values if v.strip()})
        if cleaned:
            tags[str(category)] = cleaned

    return Preset(name=name, prompt=prompt, tags=tags)


def load_presets(path: str | Path = PROMPTS_FILE) -> list[Preset]:
    """Load presets from *path*. Missing file -> empty list (not an error)."""
    file_path = Path(path)
    if not file_path.is_file():
        return []
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptError(f"{file_path} is not valid JSON: {exc}") from exc

    if isinstance(raw, dict):
        raw = raw.get("presets", [])
    if not isinstance(raw, list):
        raise PromptError(f"{file_path} must contain a list of presets "
                          "(or an object with a top-level \"presets\" list)")

    names_seen: set[str] = set()
    presets: list[Preset] = []
    for i, entry in enumerate(raw):
        preset = _validate_entry(entry, i)
        key = preset.name.strip().lower()
        if key in names_seen:
            raise PromptError(f"duplicate preset name: {preset.name!r}")
        names_seen.add(key)
        presets.append(preset)
    return presets


def find_preset(presets: list[Preset], name: str) -> Preset | None:
    target = name.strip().lower()
    for preset in presets:
        if preset.name.strip().lower() == target:
            return preset
    return None
