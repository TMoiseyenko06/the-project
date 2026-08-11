from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagebatch.config import Config  # noqa: E402
from imagebatch.storage import AlbumStore  # noqa: E402


@pytest.fixture
def store(tmp_path: Path) -> AlbumStore:
    return AlbumStore(tmp_path / "outputs")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        model_id="mock",
        output_dir=str(tmp_path / "outputs"),
        batch_size=1,
        max_side=64,
        seed=None,
    )


def make_image(path: Path, size: tuple[int, int] = (32, 32), color: str = "red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "sources"
    for i, color in enumerate(["red", "green", "blue", "yellow"]):
        make_image(directory / f"img_{i}.png", color=color)
    return directory


class HookedPipe:
    """Wraps a pipeline callable so tests can intercept calls.

    Needed because ``__call__`` is looked up on the type, so monkeypatching the
    attribute on a pipeline *instance* would never take effect.
    """

    def __init__(self, inner, hook) -> None:
        self.inner = inner
        self.hook = hook
        self.calls: list[int] = []

    def __call__(self, *args, **kwargs):
        images = kwargs.get("image")
        images = images if isinstance(images, list) else [images]
        self.calls.append(len(images))
        self.hook(images)
        return self.inner(*args, **kwargs)
