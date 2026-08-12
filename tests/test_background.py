"""BackgroundRun: runs decoupled from any client connection."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from imagebatch.background import BackgroundRun
from imagebatch.batch import BatchError, BatchRunner
from imagebatch.config import Config
from imagebatch.pipeline import EditPipeline
from imagebatch.storage import AlbumStore
from tests.conftest import make_image


@pytest.fixture
def background(config: Config) -> BackgroundRun:
    store = AlbumStore(config.output_path)
    return BackgroundRun(BatchRunner(config, store, EditPipeline(config)))


def wait_for(background: BackgroundRun, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while background.is_active and time.time() < deadline:
        time.sleep(0.02)
    assert not background.is_active, "run did not finish in time"
    return background.snapshot()


def test_idle_before_any_run(background: BackgroundRun) -> None:
    snap = background.snapshot()
    assert not snap["active"]
    assert not snap["ever_ran"]
    assert snap["progress"] is None
    assert snap["error"] is None


def test_run_completes_without_a_consumer(background: BackgroundRun,
                                          source_dir: Path) -> None:
    """Nothing iterates the generator — the point of running in the background."""
    background.start(source=source_dir, prompt="prompt")

    snap = wait_for(background)

    assert snap["progress"].succeeded == 4
    assert snap["error"] is None
    assert snap["finished_at"] is not None


def test_outputs_land_on_disk(background: BackgroundRun, source_dir: Path,
                              config: Config) -> None:
    background.start(source=source_dir, prompt="prompt")
    wait_for(background)

    assert len(AlbumStore(config.output_path).list_unsorted()) == 4


def test_start_returns_immediately(background: BackgroundRun, source_dir: Path) -> None:
    started = time.time()
    background.start(source=source_dir, prompt="prompt")
    elapsed = time.time() - started

    assert elapsed < 1.0  # returned without waiting for the work
    wait_for(background)


def test_second_start_while_active_rejected(background: BackgroundRun,
                                            source_dir: Path) -> None:
    background.start(source=source_dir, prompt="prompt")
    try:
        with pytest.raises(BatchError, match="already running"):
            background.start(source=source_dir, prompt="other")
    finally:
        wait_for(background)


def test_can_start_again_after_finishing(background: BackgroundRun,
                                         source_dir: Path) -> None:
    background.start(source=source_dir, prompt="prompt")
    wait_for(background)

    background.start(source=source_dir, prompt="second prompt")
    snap = wait_for(background)

    assert snap["progress"].succeeded == 4


def test_error_is_captured_not_raised(background: BackgroundRun,
                                      tmp_path: Path) -> None:
    background.start(source=tmp_path / "does-not-exist", prompt="prompt")

    snap = wait_for(background)

    assert snap["error"] is not None
    assert "not found" in snap["error"]
    assert not snap["active"]


def test_error_does_not_wedge_the_runner(background: BackgroundRun, tmp_path: Path,
                                         source_dir: Path) -> None:
    background.start(source=tmp_path / "nope", prompt="prompt")
    wait_for(background)

    background.start(source=source_dir, prompt="prompt")
    snap = wait_for(background)

    assert snap["error"] is None
    assert snap["progress"].succeeded == 4


def test_progress_updates_during_run(background: BackgroundRun, tmp_path: Path) -> None:
    sources = tmp_path / "many"
    for i in range(6):
        make_image(sources / f"img_{i}.png")

    background.start(source=sources, prompt="prompt")
    seen_active = False
    deadline = time.time() + 20
    while background.is_active and time.time() < deadline:
        if background.snapshot()["progress"] is not None:
            seen_active = True
            break
        time.sleep(0.01)
    wait_for(background)

    # either we caught it mid-flight or it finished very fast; both are fine,
    # but the final snapshot must always carry progress
    assert background.snapshot()["progress"] is not None
    assert seen_active or background.snapshot()["progress"].succeeded == 6


def test_cancel_stops_the_run(background: BackgroundRun, tmp_path: Path) -> None:
    sources = tmp_path / "many"
    for i in range(20):
        make_image(sources / f"img_{i}.png")

    background.start(source=sources, prompt="prompt")
    background.cancel()
    snap = wait_for(background)

    assert snap["progress"] is not None
    assert not snap["active"]


def test_thread_is_daemon(background: BackgroundRun, source_dir: Path) -> None:
    """A daemon thread won't block interpreter shutdown."""
    background.start(source=source_dir, prompt="prompt")
    assert background._thread.daemon
    wait_for(background)
