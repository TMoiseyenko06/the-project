"""Batch runner: discover sources, process them, keep the manifest up to date.

Design points that matter in practice:

* One image failing (corrupt file, OOM, model error) never kills the run — it is
  logged, recorded in the manifest and skipped.
* A batched call that fails is retried image-by-image, so one bad image in a
  group of four doesn't lose the other three.
* Progress is emitted as events, so the UI layer stays free of runner logic.
* The manifest is flushed as work completes, so a crash mid-run loses at most
  the current image.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image, UnidentifiedImageError

from .config import Config
from .manifest import Manifest, hash_file
from .pipeline import EditPipeline, OutOfMemoryError
from .storage import UNSORTED, AlbumStore, StorageError, is_image, unique_dir, unique_path

log = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
SAVE_EVERY = 5  # flush the manifest at least this often
PROBE_MIN_ITEMS = 8   # don't pay for a batch probe on a tiny run
PROBE_CANDIDATES = 5  # how many images to try before giving up on probing


class BatchError(RuntimeError):
    """Raised for user-facing problems with the batch request itself."""


@dataclass
class ImageResult:
    source: Path
    output: Path | None = None
    status: str = "ok"  # ok | failed | skipped
    error: str | None = None
    duration: float = 0.0


@dataclass
class Progress:
    """A snapshot emitted after each image (or batch) is handled."""

    done: int = 0
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    current: str = ""
    elapsed: float = 0.0
    eta: float | None = None
    finished: bool = False
    message: str = ""
    batch_size: int = 1
    results: list[ImageResult] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def describe_progress(progress: Progress) -> str:
    """One-line human-readable status for the UI."""
    if progress.total == 0:
        return progress.message or "Nothing to do."
    parts = [
        f"**{progress.done} / {progress.total}** images",
        f"✅ {progress.succeeded}",
    ]
    if progress.failed:
        parts.append(f"❌ {progress.failed}")
    if progress.skipped:
        parts.append(f"⏭️ {progress.skipped} already done")
    parts.append(f"elapsed {_format_duration(progress.elapsed)}")
    if not progress.finished:
        parts.append(f"ETA {_format_duration(progress.eta)}")
    line = " · ".join(parts)
    if progress.current and not progress.finished:
        line += f"\n\nCurrent: `{progress.current}`"
    if progress.message:
        line += f"\n\n{progress.message}"
    return line


def extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    """Extract images from a zip, ignoring paths that escape *dest*."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name.startswith(".") or not is_image(Path(name)):
                continue
            target = unique_path(dest, name)
            # Resolve and confirm containment: never trust archive member paths.
            if not str(target.resolve()).startswith(str(dest.resolve())):
                log.warning("Skipping unsafe zip member %s", info.filename)
                continue
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
    log.info("Extracted %d image(s) from %s", len(extracted), zip_path.name)
    return extracted


def stage_uploads(paths: Sequence[str | Path], staging_dir: Path) -> Path:
    """Copy individually-uploaded images into a fresh folder and return it.

    Lets a batch of separately-picked files (e.g. a phone's multi-select photo
    picker, which has no folder or zip to point at) reuse the same
    folder-processing path as ``--cli /some/folder``, instead of needing a
    second code path through the runner.
    """
    # unique_dir guards against two uploads landing in the same wall-clock
    # second merging into one folder, which would silently combine batches.
    target = unique_dir(staging_dir, f"upload_{int(time.time())}")
    staged: list[Path] = []
    for raw in paths:
        src = Path(raw)
        if not src.is_file() or not is_image(src):
            log.warning("Skipping non-image upload: %s", src.name)
            continue
        dst = unique_path(target, src.name)
        shutil.copy2(src, dst)
        staged.append(dst)
    if not staged:
        raise BatchError("no valid images were uploaded")
    log.info("Staged %d uploaded image(s) in %s", len(staged), target)
    return target


def discover_images(source: str | Path, staging_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect image paths from a folder, a zip file, or a single image."""
    path = Path(str(source)).expanduser()
    if not path.exists():
        raise BatchError(f"source not found: {path}")

    if path.is_file():
        if path.suffix.lower() == ".zip":
            target = unique_dir(staging_dir, f"zip_{int(time.time())}_{path.stem[:40]}")
            images = extract_zip(path, target)
            if not images:
                raise BatchError(f"no images found inside {path.name}")
            return sorted(images)
        if is_image(path):
            return [path]
        raise BatchError(f"unsupported file type: {path.suffix or path.name}")

    pattern = "**/*" if recursive else "*"
    images = sorted(p for p in path.glob(pattern) if p.is_file() and is_image(p))
    if not images:
        raise BatchError(f"no images found in {path}")
    return images


class BatchRunner:
    """Runs a prompt over a set of images, resumably."""

    def __init__(self, config: Config, store: AlbumStore, pipeline: EditPipeline,
                 manifest: Manifest | None = None) -> None:
        self.config = config
        self.store = store
        self.pipeline = pipeline
        # `manifest or ...` would be wrong: Manifest defines __len__, so an empty
        # one is falsy and would be silently replaced by a second instance
        # writing to the same file.
        self.manifest = (
            manifest if manifest is not None
            else Manifest(store.root / MANIFEST_FILE, root=store.root)
        )
        self._cancel = threading.Event()
        self._running = threading.Lock()
        # Set by run() for the duration of a run; None when face detection is off.
        self._face_detector: Any = None
        self._face_registry: Any = None

    # -- control ----------------------------------------------------------
    def cancel(self) -> None:
        """Ask the current run to stop after the in-flight image finishes."""
        self._cancel.set()

    @property
    def is_running(self) -> bool:
        return self._running.locked()

    # -- face recognition wiring -------------------------------------------
    # Separate factories so tests can substitute fakes without needing the
    # optional insightface dependency installed.
    def _build_face_detector(self):
        from .faces import InsightFaceDetector

        detector = InsightFaceDetector(use_gpu=self.config.face_use_gpu)
        detector.load()  # surfaces a missing-package error before any work starts
        return detector

    def _build_face_registry(self):
        from .faces import FACES_FILE, FaceRegistry

        return FaceRegistry(self.store.root / FACES_FILE,
                            match_threshold=self.config.face_match_threshold)

    # -- helpers ----------------------------------------------------------
    def _output_path(self, source: Path, target_album: str | None) -> Path:
        suffix = self.config.output_format.lower().replace("jpeg", "jpg")
        directory = self.store.dir_for(target_album)
        directory.mkdir(parents=True, exist_ok=True)
        return unique_path(directory, f"{source.stem}.{suffix}")

    def _save(self, image: Image.Image, path: Path) -> None:
        fmt = self.config.output_format.lower()
        params: dict[str, Any] = {}
        if fmt in ("jpg", "jpeg"):
            params = {"quality": self.config.output_quality, "subsampling": 0}
            image = image.convert("RGB")
            fmt = "JPEG"
        elif fmt == "webp":
            params = {"quality": self.config.output_quality}
            fmt = "WEBP"
        else:
            fmt = "PNG"
        image.save(path, format=fmt, **params)

    def _plan(self, sources: Sequence[Path], prompt: str, resume: bool,
              tags: dict[str, list[str]] | None = None) -> tuple[list[dict], int]:
        """Split sources into work items and count the ones already finished."""
        signature = self.config.signature()
        tags = tags or {}
        todo: list[dict] = []
        skipped = 0
        for source in sources:
            try:
                digest = hash_file(source)
            except OSError as exc:
                # Unreadable file: keep it in the list so it's reported as a
                # failure rather than silently vanishing from the totals.
                todo.append({"source": source, "hash": "", "key": "", "error": str(exc),
                             "tags": tags})
                continue
            key = Manifest.make_key(digest, prompt, signature)
            if resume and self.manifest.is_done(key):
                skipped += 1
                continue
            todo.append({"source": source, "hash": digest, "key": key, "error": None,
                         "tags": tags})
        return todo, skipped

    def _tune_batch_size(self, items: list[dict], prompt: str) -> int:
        """Pick a batch size, measuring single vs batched throughput if asked.

        With ``batch_size: auto`` we time one image on its own, then a small
        batch, and keep whichever gives the better per-image time. Models that
        don't benefit from batching (or don't fit) fall back to 1.
        """
        configured = self.config.batch_size
        if isinstance(configured, int):
            return max(1, configured)
        candidate = min(self.config.auto_batch_max, len(items))
        if candidate < 2:
            return 1
        if len(items) < PROBE_MIN_ITEMS:
            # Probing costs two extra inferences; on a short run that outweighs
            # anything batching could save.
            log.info("Only %d image(s) to do; skipping the batch probe", len(items))
            return 1

        probe = None
        for item in items[:PROBE_CANDIDATES]:
            try:
                with Image.open(item["source"]) as img:
                    probe = img.copy()
                break
            except Exception as exc:  # noqa: BLE001 - probing is best-effort
                log.debug("Probe candidate %s unreadable (%s)", item["source"].name, exc)
        if probe is None:
            log.warning("No readable image among the first %d; using batch_size=1",
                        PROBE_CANDIDATES)
            return 1

        log.info("Probing batch throughput (1 vs %d)...", candidate)
        try:
            started = time.time()
            self.pipeline.edit([probe], prompt)
            single = time.time() - started

            started = time.time()
            self.pipeline.edit([probe] * candidate, prompt)
            batched = (time.time() - started) / candidate
        except OutOfMemoryError:
            log.warning("Batch probe hit OOM; using batch_size=1")
            self.pipeline.free_memory()
            return 1
        except Exception as exc:  # noqa: BLE001 - never fail a run over a probe
            log.warning("Batch probe failed (%s); using batch_size=1", exc)
            return 1
        chosen = candidate if batched < single * 0.95 else 1
        log.info("Batch probe: %.2fs/img single vs %.2fs/img batched -> batch_size=%d",
                 single, batched, chosen)
        return chosen

    def _process_group(self, group: list[dict], prompt: str,
                       target_album: str | None = None) -> list[ImageResult]:
        """Process a group of items, falling back to one-by-one on failure."""
        if len(group) == 1:
            return [self._process_one(group[0], prompt, target_album)]

        loaded: list[Image.Image] = []
        usable: list[dict] = []
        results: list[ImageResult] = []
        for item in group:
            if item.get("error"):
                results.append(self._fail(item, prompt, item["error"]))
                continue
            try:
                with Image.open(item["source"]) as img:
                    loaded.append(img.copy())
                usable.append(item)
            except (OSError, UnidentifiedImageError) as exc:
                results.append(self._fail(item, prompt, f"unreadable image: {exc}"))

        if not usable:
            return results

        started = time.time()
        try:
            edited = self.pipeline.edit(loaded, prompt)
        except Exception as exc:  # noqa: BLE001 - degrade to per-image processing
            reason = "OOM" if isinstance(exc, OutOfMemoryError) else type(exc).__name__
            log.warning("Batch of %d failed (%s: %s); retrying individually",
                        len(usable), reason, exc)
            self.pipeline.free_memory()
            for item in usable:
                results.append(self._process_one(item, prompt, target_album))
            return results

        per_image = (time.time() - started) / max(1, len(usable))
        for item, image in zip(usable, edited):
            results.append(self._store_result(item, image, prompt, per_image, target_album))
        return results

    def _process_one(self, item: dict, prompt: str,
                     target_album: str | None = None) -> ImageResult:
        if item.get("error"):
            return self._fail(item, prompt, item["error"])
        source: Path = item["source"]
        started = time.time()
        log.info("  %s: loading", source.name)
        try:
            with Image.open(source) as img:
                image = img.copy()
        except (OSError, UnidentifiedImageError) as exc:
            return self._fail(item, prompt, f"unreadable image: {exc}")

        log.info("  %s: running inference (%s steps)", source.name,
                 self.config.num_inference_steps)
        try:
            edited = self.pipeline.edit([image], prompt)
            log.info("  %s: inference done in %.1fs", source.name, time.time() - started)
        except OutOfMemoryError as exc:
            self.pipeline.free_memory()
            return self._fail(item, prompt, f"out of GPU memory: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
            log.exception("Inference failed for %s", source.name)
            return self._fail(item, prompt, f"{type(exc).__name__}: {exc}")

        if not edited:
            return self._fail(item, prompt, "pipeline returned no image")
        return self._store_result(item, edited[0], prompt, time.time() - started,
                                  target_album)

    def _store_result(self, item: dict, image: Image.Image, prompt: str,
                      duration: float, target_album: str | None = None) -> ImageResult:
        source: Path = item["source"]
        try:
            output = self._output_path(source, target_album)
            self._save(image, output)
        except OSError as exc:
            return self._fail(item, prompt, f"could not write output: {exc}")
        if target_album and target_album != UNSORTED:
            # unsorted/ is read straight off disk, but album membership lives in
            # albums.json and has to be recorded.
            try:
                self.store.register_image(target_album, output.name)
            except StorageError as exc:
                log.warning("Saved %s but could not file it into %s: %s",
                            output.name, target_album, exc)
        log.info("  %s: saved -> %s", source.name, output.name)
        self._apply_tags(source, target_album, output.name, item.get("tags") or {})
        log.info("  %s: complete", source.name)
        if item.get("key"):
            self.manifest.record_success(item["key"], source, item["hash"], output,
                                         prompt, duration)
        return ImageResult(source=source, output=output, status="ok", duration=duration)

    def _apply_tags(self, source: Path, album: str | None, filename: str,
                    batch_tags: dict[str, list[str]]) -> None:
        """Tag a freshly written output: batch-level tags plus detected faces.

        Never raises — a tagging problem must not turn a successfully generated
        image into a failed one, so problems are logged and the image is kept.
        """
        tags = {category: list(values) for category, values in batch_tags.items()}
        log.info("  %s: tagging (face detection %s)", filename,
                 "on" if self._face_detector is not None else "off")

        if self._face_detector is not None:
            # Detect on the *source* image, not the edit: the edit may have
            # altered the face, and the source is what actually identifies who
            # is in the photo.
            try:
                with Image.open(source) as img:
                    detections = self._face_detector.detect(img)
                face_ids = [self._face_registry.match_or_create(d.embedding)
                            for d in detections]
                if face_ids:
                    category = self.config.face_category
                    tags.setdefault(category, [])
                    for face_id in face_ids:
                        if face_id not in tags[category]:
                            tags[category].append(face_id)
                    log.info("%s: detected %d face(s) -> %s",
                             source.name, len(face_ids), ", ".join(face_ids))
            except Exception as exc:  # noqa: BLE001 - never lose an image over tagging
                log.warning("Face detection failed for %s: %s", source.name, exc)

        for category, values in tags.items():
            if not values:
                continue
            try:
                log.debug("  %s: writing tag %s=%s", filename, category, values)
                self.store.tags.set_tags(album, filename, category, values)
            except StorageError as exc:
                log.warning("Could not tag %s with %s=%s: %s",
                            filename, category, values, exc)

    def _fail(self, item: dict, prompt: str, error: str) -> ImageResult:
        source: Path = item["source"]
        log.warning("Failed: %s (%s)", source.name, error)
        if item.get("key"):
            self.manifest.record_failure(item["key"], source, item.get("hash", ""), prompt, error)
        return ImageResult(source=source, status="failed", error=error)

    # -- the run ----------------------------------------------------------
    def run(self, source: str | Path, prompt: str, resume: bool = True,
            recursive: bool = True, target_album: str | None = None,
            tags: dict[str, list[str]] | None = None,
            detect_faces: bool = False) -> Iterator[Progress]:
        """Process every image under *source*, yielding progress as it goes.

        Results land in ``unsorted/`` unless *target_album* names an album, in
        which case every image in the batch is filed there as it is produced.

        *tags* are applied to every image the run produces. With *detect_faces*,
        each source image is additionally run through face recognition and the
        resulting face ids are added under the configured face category, so a
        folder of mixed people gets its "who" tagged automatically while "what"
        comes from the batch-level tags.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise BatchError("a prompt is required")
        if self.is_running:
            raise BatchError(
                "a batch is already running — if you refreshed the page mid-run, "
                "wait a moment and try again")
        if target_album == UNSORTED:
            target_album = None
        if target_album is not None:
            # Fail before any work happens rather than after the first image.
            try:
                self.store.get_album(target_album)
            except StorageError as exc:
                raise BatchError(str(exc)) from exc

        batch_tags = {c: list(v) for c, v in (tags or {}).items() if v}
        # Create any categories the caller referenced but that don't exist yet,
        # so a preset naming a fresh category just works.
        for category in list(batch_tags) + ([self.config.face_category] if detect_faces else []):
            try:
                self.store.tags.create_category(category)
            except StorageError:
                pass  # already exists

        if detect_faces:
            # Fail before any work happens if the optional dependency is absent.
            try:
                self._face_detector = self._build_face_detector()
                self._face_registry = self._build_face_registry()
            except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing error
                self._face_detector = self._face_registry = None
                raise BatchError(str(exc)) from exc
        else:
            self._face_detector = self._face_registry = None

        with self._running:
            self._cancel.clear()
            started = time.time()
            sources = discover_images(source, self.store.staging_dir, recursive=recursive)
            log.info("Discovered %d source image(s)", len(sources))

            log.info("Planning run (hashing %d source(s) for resume)...", len(sources))
            items, skipped = self._plan(sources, prompt, resume, batch_tags)
            log.info("Plan ready: %d to process, %d already done", len(items), skipped)
            progress = Progress(total=len(items), skipped=skipped,
                                message="Loading model..." if not self.pipeline.loaded else "")
            yield progress

            if not items:
                return_message = (
                    f"Nothing to do — all {skipped} image(s) were already processed "
                    "with this prompt and model."
                ) if skipped else "No images to process."
                yield Progress(total=0, skipped=skipped, finished=True, elapsed=time.time() - started,
                               message=return_message)
                return

            try:
                self.pipeline.load()
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                log.exception("Model failed to load")
                yield Progress(total=len(items), skipped=skipped, finished=True,
                               elapsed=time.time() - started,
                               message=f"❌ Could not load the model: {exc}")
                return

            log.info("Choosing batch size (config: %r)...", self.config.batch_size)
            batch_size = self._tune_batch_size(items, prompt)
            log.info("Batch size: %d — starting %d image(s)", batch_size, len(items))
            results: list[ImageResult] = []
            succeeded = failed = 0
            since_save = 0

            for offset in range(0, len(items), batch_size):
                if self._cancel.is_set():
                    break
                group = items[offset:offset + batch_size]
                log.info("Group %d/%d: %s", offset // batch_size + 1,
                         (len(items) + batch_size - 1) // batch_size,
                         ", ".join(i["source"].name for i in group))
                progress = Progress(
                    done=len(results), total=len(items), succeeded=succeeded, failed=failed,
                    skipped=skipped, current=group[0]["source"].name,
                    elapsed=time.time() - started, batch_size=batch_size,
                    eta=self._eta(len(results), len(items), time.time() - started),
                )
                yield progress

                group_results = self._process_group(group, prompt, target_album)
                results.extend(group_results)
                succeeded += sum(1 for r in group_results if r.status == "ok")
                failed += sum(1 for r in group_results if r.status == "failed")

                since_save += len(group_results)
                if since_save >= SAVE_EVERY:
                    self.manifest.save_if_dirty()
                    since_save = 0

                yield Progress(
                    done=len(results), total=len(items), succeeded=succeeded, failed=failed,
                    skipped=skipped, current=group[-1]["source"].name,
                    elapsed=time.time() - started, batch_size=batch_size,
                    eta=self._eta(len(results), len(items), time.time() - started),
                )

            self.manifest.save_if_dirty()
            elapsed = time.time() - started
            cancelled = self._cancel.is_set()
            summary = self._summarise(results, skipped, elapsed, cancelled, batch_size,
                                      target_album)
            self._write_run_log(prompt, results, skipped, elapsed, cancelled, target_album)

            yield Progress(
                done=len(results), total=len(items), succeeded=succeeded, failed=failed,
                skipped=skipped, elapsed=elapsed, finished=True, batch_size=batch_size,
                message=summary, results=results,
            )

    @staticmethod
    def _eta(done: int, total: int, elapsed: float) -> float | None:
        if done <= 0:
            return None
        return (elapsed / done) * (total - done)

    def _summarise(self, results: list[ImageResult], skipped: int, elapsed: float,
                   cancelled: bool, batch_size: int, target_album: str | None = None) -> str:
        succeeded = [r for r in results if r.status == "ok"]
        failures = [r for r in results if r.status == "failed"]
        head = "🛑 **Cancelled**" if cancelled else "🏁 **Finished**"
        lines = [
            f"{head} in {_format_duration(elapsed)} "
            f"(batch size {batch_size}"
            + (f", {elapsed / len(results):.1f}s per image" if results else "")
            + ")",
            "",
            f"- ✅ Processed: **{len(succeeded)}**",
            f"- ❌ Failed: **{len(failures)}**",
            f"- ⏭️ Skipped (already done): **{skipped}**",
        ]
        if succeeded:
            if target_album:
                where = self.store.path_name(target_album)
                lines.append(f"\nResults were filed into **{where}** — "
                             "open the **Gallery** tab to review them.")
            else:
                lines.append(f"\nResults are in `{self.store.unsorted_dir}` — "
                             "open the **Gallery** tab to review and file them.")
        if failures:
            lines.append("\n**Failures**\n")
            lines.append("| Image | Reason |")
            lines.append("| --- | --- |")
            for result in failures[:25]:
                reason = (result.error or "unknown").replace("|", "\\|")[:160]
                lines.append(f"| `{result.source.name}` | {reason} |")
            if len(failures) > 25:
                lines.append(f"\n_...and {len(failures) - 25} more; see the run log._")
        return "\n".join(lines)

    def _write_run_log(self, prompt: str, results: list[ImageResult], skipped: int,
                       elapsed: float, cancelled: bool,
                       target_album: str | None = None) -> Path:
        path = self.store.logs_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "prompt": prompt,
            "target_album": target_album,
            "model": self.config.model_id,
            "pipeline_class": self.config.pipeline_class,
            "settings": self.config.signature(),
            "elapsed_seconds": round(elapsed, 2),
            "cancelled": cancelled,
            "skipped": skipped,
            "succeeded": sum(1 for r in results if r.status == "ok"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "results": [
                {
                    "source": str(r.source),
                    "output": str(r.output) if r.output else None,
                    "status": r.status,
                    "error": r.error,
                    "seconds": round(r.duration, 2),
                }
                for r in results
            ],
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not write run log: %s", exc)
        return path
