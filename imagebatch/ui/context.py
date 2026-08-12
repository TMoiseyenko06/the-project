"""Shared application state handed to every tab."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..background import BackgroundRun
from ..batch import MANIFEST_FILE, BatchRunner
from ..config import Config
from ..faces import FACES_FILE, FaceRegistry
from ..manifest import Manifest
from ..pipeline import EditPipeline
from ..prompts import Preset, PromptError, load_presets
from ..storage import AlbumStore
from ..thumbnails import ThumbnailCache

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    config: Config
    store: AlbumStore
    pipeline: EditPipeline
    runner: BatchRunner
    manifest: Manifest
    thumbnails: ThumbnailCache
    faces: FaceRegistry
    background: BackgroundRun
    presets: list[Preset]
    preset_error: str | None = None

    @classmethod
    def create(cls, config: Config) -> "AppContext":
        store = AlbumStore(config.output_path)
        # Reconcile albums.json with disk in case files were moved by hand.
        store.sync()
        manifest = Manifest(store.root / MANIFEST_FILE, root=store.root)
        pipeline = EditPipeline(config)
        runner = BatchRunner(config, store, pipeline, manifest=manifest)
        thumbnails = ThumbnailCache(store.staging_dir / "thumbs", size=config.thumbnail_size)
        faces = FaceRegistry(store.root / FACES_FILE,
                             match_threshold=config.face_match_threshold)
        background = BackgroundRun(runner)

        # A broken prompts.json shouldn't stop the app from starting — surface
        # it in the UI and carry on with no presets.
        presets: list[Preset] = []
        preset_error: str | None = None
        try:
            presets = load_presets()
        except PromptError as exc:
            preset_error = str(exc)
            log.warning("Could not load prompt presets: %s", exc)

        return cls(config=config, store=store, pipeline=pipeline, runner=runner,
                   manifest=manifest, thumbnails=thumbnails, faces=faces,
                   background=background, presets=presets, preset_error=preset_error)

    def reload_presets(self) -> str | None:
        """Re-read prompts.json so edits land without an app restart."""
        try:
            self.presets = load_presets()
            self.preset_error = None
        except PromptError as exc:
            self.preset_error = str(exc)
        return self.preset_error
