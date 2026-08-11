from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from imagebatch.batch import BatchError, BatchRunner, discover_images, extract_zip
from imagebatch.config import Config
from imagebatch.manifest import Manifest
from imagebatch.pipeline import EditPipeline
from imagebatch.storage import AlbumStore
from tests.conftest import HookedPipe, make_image


def build_runner(config: Config, store: AlbumStore) -> BatchRunner:
    return BatchRunner(config, store, EditPipeline(config))


def run_to_completion(runner: BatchRunner, source, prompt: str, **kwargs):
    last = None
    for progress in runner.run(source, prompt, **kwargs):
        last = progress
    assert last is not None and last.finished
    return last


# -- discovery ------------------------------------------------------------
def test_discover_folder(source_dir: Path, tmp_path: Path) -> None:
    found = discover_images(source_dir, tmp_path / "staging")
    assert len(found) == 4
    assert all(p.suffix == ".png" for p in found)


def test_discover_ignores_non_images(source_dir: Path, tmp_path: Path) -> None:
    (source_dir / "notes.txt").write_text("hello")
    assert len(discover_images(source_dir, tmp_path / "staging")) == 4


def test_discover_recursive(source_dir: Path, tmp_path: Path) -> None:
    make_image(source_dir / "nested" / "deep.png")
    assert len(discover_images(source_dir, tmp_path / "staging")) == 5
    assert len(discover_images(source_dir, tmp_path / "staging", recursive=False)) == 4


def test_discover_missing_source(tmp_path: Path) -> None:
    with pytest.raises(BatchError):
        discover_images(tmp_path / "nope", tmp_path / "staging")


def test_discover_empty_folder(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BatchError):
        discover_images(empty, tmp_path / "staging")


def test_discover_zip(source_dir: Path, tmp_path: Path) -> None:
    archive = tmp_path / "batch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for image in source_dir.glob("*.png"):
            zf.write(image, arcname=image.name)

    found = discover_images(archive, tmp_path / "staging")

    assert len(found) == 4


def test_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "evil.png"
    make_image(payload)
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(payload, arcname="../../escaped.png")
        zf.write(payload, arcname="fine.png")

    extracted = extract_zip(archive, tmp_path / "staging")

    assert [p.name for p in extracted] == ["escaped.png", "fine.png"]
    assert all((tmp_path / "staging") in p.parents for p in extracted)
    assert not (tmp_path.parent / "escaped.png").exists()


# -- running --------------------------------------------------------------
def test_run_processes_all_images(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    final = run_to_completion(runner, source_dir, "make it pop")

    assert final.succeeded == 4
    assert final.failed == 0
    assert len(store.list_unsorted()) == 4
    assert "Processed: **4**" in final.message


def test_outputs_land_in_unsorted(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    run_to_completion(build_runner(config, store), source_dir, "prompt")

    assert sorted(store.list_unsorted()) == [f"img_{i}.png" for i in range(4)]
    assert not any(store.root.glob("album_*"))


def test_progress_events_are_monotonic(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    events = list(runner.run(source_dir, "prompt"))

    done = [e.done for e in events]
    assert done == sorted(done)
    assert events[-1].done == 4
    assert events[-1].fraction == 1.0


def test_resume_skips_finished_images(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")

    second = run_to_completion(runner, source_dir, "prompt")

    assert second.skipped == 4
    assert second.succeeded == 0
    assert len(store.list_unsorted()) == 4  # no duplicates written


def test_resume_reprocesses_new_images(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")
    make_image(source_dir / "img_new.png", color="purple")

    second = run_to_completion(runner, source_dir, "prompt")

    assert second.succeeded == 1
    assert second.skipped == 4
    assert len(store.list_unsorted()) == 5


def test_changing_prompt_reprocesses(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt one")

    second = run_to_completion(runner, source_dir, "prompt two")

    assert second.succeeded == 4
    assert second.skipped == 0
    assert len(store.list_unsorted()) == 8


def test_resume_disabled_reprocesses(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")

    second = run_to_completion(runner, source_dir, "prompt", resume=False)

    assert second.succeeded == 4
    assert second.skipped == 0


def test_resume_survives_moving_output_to_album(config: Config, source_dir: Path) -> None:
    """Filing results into an album must not make them look unprocessed."""
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")
    album = store.create_album("Keepers")
    store.assign(store.list_unsorted(), None, album.slug)

    second = run_to_completion(runner, source_dir, "prompt")

    assert second.skipped == 4
    assert second.succeeded == 0


def test_deleting_output_causes_reprocess(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")
    store.delete_images(["img_0.png"], None)

    second = run_to_completion(runner, source_dir, "prompt")

    assert second.succeeded == 1
    assert second.skipped == 3


def test_corrupt_image_is_reported_not_fatal(config: Config, source_dir: Path) -> None:
    (source_dir / "broken.png").write_bytes(b"this is not a png")
    store = AlbumStore(config.output_path)

    final = run_to_completion(build_runner(config, store), source_dir, "prompt")

    assert final.succeeded == 4
    assert final.failed == 1
    assert "broken.png" in final.message
    assert len(store.list_unsorted()) == 4


def test_inference_failure_is_isolated(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    runner.pipeline.load()

    def explode_on_green(images):
        if any(img.getpixel((0, 0)) == (0, 128, 0) for img in images):  # exactly "green"
            raise RuntimeError("simulated model explosion")

    runner.pipeline.pipe = HookedPipe(runner.pipeline.pipe, explode_on_green)

    final = run_to_completion(runner, source_dir, "prompt")

    assert final.failed == 1
    assert final.succeeded == 3
    assert "simulated model explosion" in final.message


def test_failed_images_retry_on_next_run(config: Config, source_dir: Path) -> None:
    broken = source_dir / "broken.png"
    broken.write_bytes(b"not an image")
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")

    make_image(broken, color="orange")  # user fixes the file
    second = run_to_completion(runner, source_dir, "prompt")

    assert second.succeeded == 1
    assert second.failed == 0


def test_empty_prompt_rejected(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    with pytest.raises(BatchError):
        list(runner.run(source_dir, "   "))


def test_run_writes_log(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    run_to_completion(build_runner(config, store), source_dir, "prompt")

    logs = list(store.logs_dir.glob("run_*.json"))
    assert len(logs) == 1
    assert "prompt" in logs[0].read_text()


def test_cancel_stops_run(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    final = None
    for i, progress in enumerate(runner.run(source_dir, "prompt")):
        if i == 2:
            runner.cancel()
        final = progress

    assert final is not None and final.finished
    assert "Cancelled" in final.message
    assert final.succeeded < 4


def test_cancelled_work_resumes(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    for i, _ in enumerate(runner.run(source_dir, "prompt")):
        if i == 2:
            runner.cancel()

    done_before = len(store.list_unsorted())
    second = run_to_completion(runner, source_dir, "prompt")

    assert second.skipped == done_before
    assert len(store.list_unsorted()) == 4


def test_manifest_written_to_output_root(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    run_to_completion(build_runner(config, store), source_dir, "prompt")

    assert (store.root / "manifest.json").is_file()


def test_batched_run_matches_single(config: Config, source_dir: Path, tmp_path: Path) -> None:
    single_store = AlbumStore(tmp_path / "single")
    run_to_completion(build_runner(config, single_store), source_dir, "prompt")

    batched_config = Config(**{**config.__dict__, "output_dir": str(tmp_path / "batched"),
                               "batch_size": 3})
    batched_store = AlbumStore(batched_config.output_path)
    final = run_to_completion(build_runner(batched_config, batched_store), source_dir, "prompt")

    assert final.succeeded == 4
    assert sorted(batched_store.list_unsorted()) == sorted(single_store.list_unsorted())
    for name in single_store.list_unsorted():
        with Image.open(single_store.unsorted_dir / name) as a, \
             Image.open(batched_store.unsorted_dir / name) as b:
            assert a.tobytes() == b.tobytes()


def test_batch_failure_falls_back_to_individual(config: Config, source_dir: Path) -> None:
    batched = Config(**{**config.__dict__, "batch_size": 4})
    store = AlbumStore(batched.output_path)
    runner = build_runner(batched, store)
    runner.pipeline.load()

    def fail_on_batch(images):
        if len(images) > 1:
            raise RuntimeError("simulated batch OOM")

    hooked = HookedPipe(runner.pipeline.pipe, fail_on_batch)
    runner.pipeline.pipe = hooked

    final = run_to_completion(runner, source_dir, "prompt")

    assert max(hooked.calls) > 1  # a batched call was attempted
    assert final.succeeded == 4  # and every image still made it through
    assert final.failed == 0


def test_zip_source_end_to_end(config: Config, source_dir: Path, tmp_path: Path) -> None:
    archive = tmp_path / "batch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for image in sorted(source_dir.glob("*.png")):
            zf.write(image, arcname=image.name)
    store = AlbumStore(config.output_path)

    final = run_to_completion(build_runner(config, store), archive, "prompt")

    assert final.succeeded == 4
    assert len(store.list_unsorted()) == 4


def test_output_format_jpg(config: Config, source_dir: Path) -> None:
    jpg_config = Config(**{**config.__dict__, "output_format": "jpg"})
    store = AlbumStore(jpg_config.output_path)

    run_to_completion(build_runner(jpg_config, store), source_dir, "prompt")

    names = store.list_unsorted()
    assert all(n.endswith(".jpg") for n in names)
    with Image.open(store.unsorted_dir / names[0]) as img:
        assert img.format == "JPEG"


def test_concurrent_run_rejected(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    generator = runner.run(source_dir, "prompt")
    next(generator)  # enter the run and take the lock

    with pytest.raises(BatchError, match="already running"):
        list(runner.run(source_dir, "other prompt"))

    generator.close()


def test_manifest_survives_corruption(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt")
    (store.root / "manifest.json").write_text("{ truncated json")

    reloaded = Manifest(store.root / "manifest.json", root=store.root)

    assert len(reloaded) == 0
    assert list(store.root.glob("manifest.corrupt.*.json"))


# -- auto batch sizing ----------------------------------------------------
def test_auto_batch_skips_probe_on_small_runs(config: Config, source_dir: Path) -> None:
    auto = Config(**{**config.__dict__, "batch_size": "auto"})
    store = AlbumStore(auto.output_path)
    runner = build_runner(auto, store)

    final = run_to_completion(runner, source_dir, "prompt")

    assert final.batch_size == 1  # only 4 images: probing would cost more than it saves
    assert final.succeeded == 4


def test_auto_batch_probes_larger_runs(config: Config, tmp_path: Path) -> None:
    sources = tmp_path / "many"
    for i in range(12):
        make_image(sources / f"img_{i:02d}.png")
    auto = Config(**{**config.__dict__, "batch_size": "auto", "auto_batch_max": 4})
    store = AlbumStore(auto.output_path)

    final = run_to_completion(build_runner(auto, store), sources, "prompt")

    assert final.succeeded == 12
    assert final.batch_size >= 1


def test_auto_batch_survives_corrupt_leading_files(config: Config, tmp_path: Path) -> None:
    """A corrupt first file must not force the probe to give up."""
    sources = tmp_path / "many"
    sources.mkdir(parents=True)
    (sources / "aaa_broken.png").write_bytes(b"not an image")
    for i in range(12):
        make_image(sources / f"img_{i:02d}.png")
    auto = Config(**{**config.__dict__, "batch_size": "auto", "auto_batch_max": 4})
    store = AlbumStore(auto.output_path)

    final = run_to_completion(build_runner(auto, store), sources, "prompt")

    assert final.succeeded == 12
    assert final.failed == 1
