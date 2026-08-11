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
from .storage import AlbumStore, is_image, unique_path

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


def discover_images(source: str | Path, staging_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect image paths from a folder, a zip file, or a single image."""
    path = Path(str(source)).expanduser()
    if not path.exists():
        raise BatchError(f"source not found: {path}")

    if path.is_file():
        if path.suffix.lower() == ".zip":
            target = staging_dir / f"zip_{int(time.time())}_{path.stem[:40]}"
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

    # -- control ----------------------------------------------------------
    def cancel(self) -> None:
        """Ask the current run to stop after the in-flight image finishes."""
        self._cancel.set()

    @property
    def is_running(self) -> bool:
        return self._running.locked()

    # -- helpers ----------------------------------------------------------
    def _output_path(self, source: Path) -> Path:
        suffix = self.config.output_format.lower().replace("jpeg", "jpg")
        return unique_path(self.store.unsorted_dir, f"{source.stem}.{suffix}")

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

    def _plan(self, sources: Sequence[Path], prompt: str, resume: bool) -> tuple[list[dict], int]:
        """Split sources into work items and count the ones already finished."""
        signature = self.config.signature()
        todo: list[dict] = []
        skipped = 0
        for source in sources:
            try:
                digest = hash_file(source)
            except OSError as exc:
                # Unreadable file: keep it in the list so it's reported as a
                # failure rather than silently vanishing from the totals.
                todo.append({"source": source, "hash": "", "key": "", "error": str(exc)})
                continue
            key = Manifest.make_key(digest, prompt, signature)
            if resume and self.manifest.is_done(key):
                skipped += 1
                continue
            todo.append({"source": source, "hash": digest, "key": key, "error": None})
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

    def _process_group(self, group: list[dict], prompt: str) -> list[ImageResult]:
        """Process a group of items, falling back to one-by-one on failure."""
        if len(group) == 1:
            return [self._process_one(group[0], prompt)]

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
                results.append(self._process_one(item, prompt))
            return results

        per_image = (time.time() - started) / max(1, len(usable))
        for item, image in zip(usable, edited):
            results.append(self._store_result(item, image, prompt, per_image))
        return results

    def _process_one(self, item: dict, prompt: str) -> ImageResult:
        if item.get("error"):
            return self._fail(item, prompt, item["error"])
        source: Path = item["source"]
        started = time.time()
        try:
            with Image.open(source) as img:
                image = img.copy()
        except (OSError, UnidentifiedImageError) as exc:
            return self._fail(item, prompt, f"unreadable image: {exc}")

        try:
            edited = self.pipeline.edit([image], prompt)
        except OutOfMemoryError as exc:
            self.pipeline.free_memory()
            return self._fail(item, prompt, f"out of GPU memory: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
            log.exception("Inference failed for %s", source.name)
            return self._fail(item, prompt, f"{type(exc).__name__}: {exc}")

        if not edited:
            return self._fail(item, prompt, "pipeline returned no image")
        return self._store_result(item, edited[0], prompt, time.time() - started)

    def _store_result(self, item: dict, image: Image.Image, prompt: str,
                      duration: float) -> ImageResult:
        source: Path = item["source"]
        try:
            output = self._output_path(source)
            self._save(image, output)
        except OSError as exc:
            return self._fail(item, prompt, f"could not write output: {exc}")
        if item.get("key"):
            self.manifest.record_success(item["key"], source, item["hash"], output,
                                         prompt, duration)
        return ImageResult(source=source, output=output, status="ok", duration=duration)

    def _fail(self, item: dict, prompt: str, error: str) -> ImageResult:
        source: Path = item["source"]
        log.warning("Failed: %s (%s)", source.name, error)
        if item.get("key"):
            self.manifest.record_failure(item["key"], source, item.get("hash", ""), prompt, error)
        return ImageResult(source=source, status="failed", error=error)

    # -- the run ----------------------------------------------------------
    def run(self, source: str | Path, prompt: str, resume: bool = True,
            recursive: bool = True) -> Iterator[Progress]:
        """Process every image under *source*, yielding progress as it goes."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise BatchError("a prompt is required")
        if self.is_running:
            raise BatchError("a batch is already running")

        with self._running:
            self._cancel.clear()
            started = time.time()
            sources = discover_images(source, self.store.staging_dir, recursive=recursive)
            log.info("Discovered %d source image(s)", len(sources))

            items, skipped = self._plan(sources, prompt, resume)
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

            batch_size = self._tune_batch_size(items, prompt)
            results: list[ImageResult] = []
            succeeded = failed = 0
            since_save = 0

            for offset in range(0, len(items), batch_size):
                if self._cancel.is_set():
                    break
                group = items[offset:offset + batch_size]
                progress = Progress(
                    done=len(results), total=len(items), succeeded=succeeded, failed=failed,
                    skipped=skipped, current=group[0]["source"].name,
                    elapsed=time.time() - started, batch_size=batch_size,
                    eta=self._eta(len(results), len(items), time.time() - started),
                )
                yield progress

                group_results = self._process_group(group, prompt)
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
            summary = self._summarise(results, skipped, elapsed, cancelled, batch_size)
            self._write_run_log(prompt, results, skipped, elapsed, cancelled)

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
                   cancelled: bool, batch_size: int) -> str:
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
                       elapsed: float, cancelled: bool) -> Path:
        path = self.store.logs_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "prompt": prompt,
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
