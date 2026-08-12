"""Run a batch on a background thread, decoupled from any browser connection.

The batch runner yields progress, which suited driving it straight from a
Gradio generator — but that tied the run's lifetime to the client: a refresh or
a navigation abandoned the generator and, with it, the run.

:class:`BackgroundRun` consumes the generator on a daemon thread instead and
keeps only the latest :class:`~imagebatch.batch.Progress` around. The UI polls
:meth:`snapshot` rather than driving the run, so closing the page (or the page
refreshing itself) leaves the work untouched — images keep landing in the
output directory and can be reviewed whenever.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .batch import BatchError, BatchRunner, Progress

log = logging.getLogger(__name__)


class BackgroundRun:
    """Owns at most one in-flight batch, and the last progress it reported."""

    def __init__(self, runner: BatchRunner) -> None:
        self.runner = runner
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._progress: Progress | None = None
        self._error: str | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None

    # -- state ---------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        """Everything the UI needs to render, captured atomically."""
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            return {
                "active": active,
                "progress": self._progress,
                "error": self._error,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "ever_ran": self._started_at is not None,
            }

    # -- control -------------------------------------------------------------
    def start(self, **run_kwargs: Any) -> None:
        """Begin a run on a daemon thread. Raises if one is already going."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise BatchError("a batch is already running")
            self._progress = None
            self._error = None
            self._finished_at = None
            self._started_at = time.time()
            self._thread = threading.Thread(
                target=self._consume, kwargs=run_kwargs,
                name="imagebatch-run", daemon=True)
            self._thread.start()
        log.info("Batch started in the background — it will keep going if you "
                 "close or refresh the page")

    def cancel(self) -> None:
        self.runner.cancel()

    def _consume(self, **run_kwargs: Any) -> None:
        try:
            for progress in self.runner.run(**run_kwargs):
                with self._lock:
                    self._progress = progress
        except BatchError as exc:
            log.warning("Batch failed: %s", exc)
            with self._lock:
                self._error = str(exc)
        except Exception as exc:  # noqa: BLE001 - thread must never die silently
            log.exception("Batch crashed")
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._finished_at = time.time()
            log.info("Background batch finished")
