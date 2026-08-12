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


# -- auto-assign to an album ----------------------------------------------
def test_results_land_in_target_album(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    album = store.create_album("Batch Output")
    runner = build_runner(config, store)

    final = run_to_completion(runner, source_dir, "prompt", target_album=album.slug)

    assert final.succeeded == 4
    assert store.list_unsorted() == []
    assert sorted(store.list_images(album.slug)) == [f"img_{i}.png" for i in range(4)]
    assert "filed into **Batch Output**" in final.message


def test_target_album_registered_in_albums_json(config: Config, source_dir: Path) -> None:
    """Files must be recorded in albums.json, not just written to the folder."""
    store = AlbumStore(config.output_path)
    album = store.create_album("Batch Output")
    run_to_completion(build_runner(config, store), source_dir, "prompt",
                      target_album=album.slug)

    reopened = AlbumStore(config.output_path)

    assert len(reopened.get_album(album.slug).images) == 4


def test_target_album_works_with_nesting(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    parent = store.create_album("Shoot")
    child = store.create_album("Edited", parent=parent.slug)

    run_to_completion(build_runner(config, store), source_dir, "prompt",
                      target_album=child.slug)

    assert len(store.list_images(child.slug)) == 4
    assert store.list_images(parent.slug) == []
    assert store.count_images(parent.slug, include_descendants=True) == 4


def test_target_album_batched(config: Config, source_dir: Path) -> None:
    batched = Config(**{**config.__dict__, "batch_size": 4})
    store = AlbumStore(batched.output_path)
    album = store.create_album("Batch Output")

    run_to_completion(build_runner(batched, store), source_dir, "prompt",
                      target_album=album.slug)

    assert len(store.list_images(album.slug)) == 4


def test_unknown_target_album_rejected(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    with pytest.raises(BatchError, match="no such album"):
        list(runner.run(source_dir, "prompt", target_album="ghost"))

    assert store.list_unsorted() == []  # nothing was written


def test_unsorted_target_behaves_like_default(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)

    run_to_completion(build_runner(config, store), source_dir, "prompt",
                      target_album="unsorted")

    assert len(store.list_unsorted()) == 4


def test_resume_works_with_target_album(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    album = store.create_album("Batch Output")
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt", target_album=album.slug)

    second = run_to_completion(runner, source_dir, "prompt", target_album=album.slug)

    assert second.skipped == 4
    assert second.succeeded == 0
    assert len(store.list_images(album.slug)) == 4  # no duplicates


def test_failures_still_isolated_with_target_album(config: Config, source_dir: Path) -> None:
    (source_dir / "broken.png").write_bytes(b"not an image")
    store = AlbumStore(config.output_path)
    album = store.create_album("Batch Output")

    final = run_to_completion(build_runner(config, store), source_dir, "prompt",
                              target_album=album.slug)

    assert final.succeeded == 4
    assert final.failed == 1
    assert len(store.list_images(album.slug)) == 4


def test_run_log_records_target_album(config: Config, source_dir: Path) -> None:
    import json

    store = AlbumStore(config.output_path)
    album = store.create_album("Batch Output")
    run_to_completion(build_runner(config, store), source_dir, "prompt",
                      target_album=album.slug)

    log_file = next(iter(store.logs_dir.glob("run_*.json")))

    assert json.loads(log_file.read_text())["target_album"] == album.slug


def test_resume_ignores_target_album(config: Config, source_dir: Path) -> None:
    """Documented behaviour: the destination is not part of the resume key."""
    store = AlbumStore(config.output_path)
    first = store.create_album("First")
    second = store.create_album("Second")
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt", target_album=first.slug)

    again = run_to_completion(runner, source_dir, "prompt", target_album=second.slug)

    assert again.skipped == 4
    assert again.succeeded == 0
    assert len(store.list_images(second.slug)) == 0  # no second copy


def test_no_resume_writes_to_the_new_album(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    first = store.create_album("First")
    second = store.create_album("Second")
    runner = build_runner(config, store)
    run_to_completion(runner, source_dir, "prompt", target_album=first.slug)

    again = run_to_completion(runner, source_dir, "prompt", target_album=second.slug,
                              resume=False)

    assert again.succeeded == 4
    assert len(store.list_images(second.slug)) == 4


# -- individual image uploads ----------------------------------------------
def test_stage_uploads_copies_into_fresh_folder(tmp_path: Path) -> None:
    from imagebatch.batch import stage_uploads

    uploads = tmp_path / "phone"
    paths = [make_image(uploads / f"IMG_{i}.jpg") for i in range(3)]
    staging = tmp_path / "staging"

    folder = stage_uploads(paths, staging)

    assert folder.parent == staging
    assert sorted(p.name for p in folder.iterdir()) == ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]
    # originals must be untouched (copy, not move) — a browser upload's temp
    # file may be needed elsewhere or cleaned up independently
    for path in paths:
        assert path.is_file()


def test_stage_uploads_skips_non_images(tmp_path: Path) -> None:
    from imagebatch.batch import stage_uploads

    uploads = tmp_path / "phone"
    good = make_image(uploads / "a.jpg")
    bad = uploads / "notes.txt"
    bad.write_text("hello")
    staging = tmp_path / "staging"

    folder = stage_uploads([good, bad], staging)

    assert [p.name for p in folder.iterdir()] == ["a.jpg"]


def test_stage_uploads_rejects_empty_result(tmp_path: Path) -> None:
    from imagebatch.batch import stage_uploads

    bad = tmp_path / "notes.txt"
    bad.write_text("hello")

    with pytest.raises(BatchError, match="no valid images"):
        stage_uploads([bad], tmp_path / "staging")


def test_stage_uploads_handles_name_collisions(tmp_path: Path) -> None:
    """Two uploaded files sharing a name (e.g. both called IMG_0.jpg) both survive."""
    from imagebatch.batch import stage_uploads

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    a = make_image(dir_a / "IMG_0.jpg", color="red")
    b = make_image(dir_b / "IMG_0.jpg", color="blue")
    staging = tmp_path / "staging"

    folder = stage_uploads([a, b], staging)

    assert len(list(folder.iterdir())) == 2


def test_uploaded_batch_runs_end_to_end(config: Config, tmp_path: Path) -> None:
    from imagebatch.batch import stage_uploads

    uploads = tmp_path / "phone"
    paths = [make_image(uploads / f"IMG_{i}.jpg") for i in range(3)]
    store = AlbumStore(config.output_path)
    folder = stage_uploads(paths, store.staging_dir)

    final = run_to_completion(build_runner(config, store), folder, "prompt")

    assert final.succeeded == 3
    assert len(store.list_unsorted()) == 3


def test_uploaded_batch_resumes_on_content(config: Config, tmp_path: Path) -> None:
    """Re-uploading the same photos (new temp paths, same bytes) still dedupes."""
    from imagebatch.batch import stage_uploads

    uploads = tmp_path / "phone"
    paths = [make_image(uploads / f"IMG_{i}.jpg", color="red") for i in range(2)]
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    first_folder = stage_uploads(paths, store.staging_dir)
    run_to_completion(runner, first_folder, "prompt")

    # Simulate a second upload of the identical photos from the phone: same
    # bytes, brand new temp filenames/paths as a browser would produce.
    reupload = tmp_path / "phone2"
    same_content_paths = [make_image(reupload / f"photo_{i}.jpg", color="red")
                          for i in range(2)]
    second_folder = stage_uploads(same_content_paths, store.staging_dir)

    second = run_to_completion(runner, second_folder, "prompt")

    assert second.skipped == 2
    assert second.succeeded == 0


# -- tagging and face detection -------------------------------------------
class FakeDetector:
    """Stands in for InsightFaceDetector; returns preset embeddings per image."""

    def __init__(self, embeddings_by_name: dict[str, list[list[float]]]) -> None:
        self.embeddings_by_name = embeddings_by_name
        self.seen: list[str] = []

    def detect(self, image):
        from imagebatch.faces import Detection
        # PIL keeps the source path on .filename when opened from disk.
        name = Path(getattr(image, "filename", "") or "").name
        self.seen.append(name)
        return [Detection(embedding=e, bbox=(0.0, 0.0, 1.0, 1.0))
                for e in self.embeddings_by_name.get(name, [])]


def vec(seed: int, dims: int = 128) -> list[float]:
    import random
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dims)]


def install_fake_faces(runner: BatchRunner, embeddings_by_name) -> FakeDetector:
    from imagebatch.faces import FaceRegistry
    detector = FakeDetector(embeddings_by_name)
    runner._build_face_detector = lambda: detector
    runner._build_face_registry = lambda: FaceRegistry(
        runner.store.root / "faces.json", match_threshold=runner.config.face_match_threshold)
    return detector


def test_batch_tags_applied_to_every_output(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    final = run_to_completion(runner, source_dir, "prompt",
                              tags={"pose": ["Standing"]})

    assert final.succeeded == 4
    for name in store.list_unsorted():
        assert store.tags.get_tags(None, name) == {"pose": ["Standing"]}


def test_batch_tags_create_missing_category(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    run_to_completion(runner, source_dir, "prompt", tags={"brand-new": ["x"]})

    assert "brand-new" in [c["key"] for c in store.tags.list_categories()]


def test_no_tags_leaves_images_untagged(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    run_to_completion(build_runner(config, store), source_dir, "prompt")
    assert store.tags.all_tagged_images() == {}


def test_face_detection_tags_person(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    # img_0 and img_2 are the same person; img_1 is someone else; img_3 has no face.
    alex, jordan = vec(1), vec(2)
    install_fake_faces(runner, {
        "img_0.png": [alex],
        "img_1.png": [jordan],
        "img_2.png": [alex],
        "img_3.png": [],
    })

    run_to_completion(runner, source_dir, "prompt", detect_faces=True)

    tags = {name: store.tags.get_tags(None, name) for name in store.list_unsorted()}
    id_0 = tags["img_0.png"]["person"][0]
    assert tags["img_2.png"]["person"] == [id_0]          # same person, same id
    assert tags["img_1.png"]["person"] != [id_0]          # different person
    assert "img_3.png" not in store.tags.all_tagged_images()  # no face, no tag


def test_face_detection_combines_with_batch_tags(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    install_fake_faces(runner, {"img_0.png": [vec(1)]})

    run_to_completion(runner, source_dir, "prompt",
                      tags={"pose": ["Sitting"]}, detect_faces=True)

    tagged = store.tags.get_tags(None, "img_0.png")
    assert tagged["pose"] == ["Sitting"]
    assert len(tagged["person"]) == 1


def test_multiple_faces_in_one_image(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    install_fake_faces(runner, {"img_0.png": [vec(1), vec(2)]})

    run_to_completion(runner, source_dir, "prompt", detect_faces=True)

    assert len(store.tags.get_tags(None, "img_0.png")["person"]) == 2


def test_face_detection_runs_on_source_not_output(config: Config, source_dir: Path) -> None:
    """Detection must see the original, since the edit can alter the face."""
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)
    detector = install_fake_faces(runner, {"img_0.png": [vec(1)]})

    run_to_completion(runner, source_dir, "prompt", detect_faces=True)

    assert sorted(detector.seen) == [f"img_{i}.png" for i in range(4)]


def test_face_detection_failure_does_not_lose_image(config: Config,
                                                    source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    class ExplodingDetector:
        def detect(self, image):
            raise RuntimeError("detector exploded")

    from imagebatch.faces import FaceRegistry
    runner._build_face_detector = lambda: ExplodingDetector()
    runner._build_face_registry = lambda: FaceRegistry(store.root / "faces.json")

    final = run_to_completion(runner, source_dir, "prompt", detect_faces=True)

    assert final.succeeded == 4  # images still produced despite tagging failing
    assert final.failed == 0


def test_missing_insightface_fails_before_any_work(config: Config,
                                                   source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    runner = build_runner(config, store)

    def explode():
        from imagebatch.faces import FaceDetectionError
        raise FaceDetectionError("insightface is not installed")

    runner._build_face_detector = explode

    with pytest.raises(BatchError, match="insightface"):
        list(runner.run(source_dir, "prompt", detect_faces=True))

    assert store.list_unsorted() == []  # nothing was processed


def test_tags_applied_when_targeting_album(config: Config, source_dir: Path) -> None:
    store = AlbumStore(config.output_path)
    album = store.create_album("Batch")
    runner = build_runner(config, store)

    run_to_completion(runner, source_dir, "prompt", target_album=album.slug,
                      tags={"pose": ["Standing"]})

    for name in store.list_images(album.slug):
        assert store.tags.get_tags(album.slug, name) == {"pose": ["Standing"]}
