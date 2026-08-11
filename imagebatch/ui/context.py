"""Shared application state handed to every tab."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..batch import MANIFEST_FILE, BatchRunner
from ..config import Config
from ..manifest import Manifest
from ..pipeline import EditPipeline
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

    @classmethod
    def create(cls, config: Config) -> "AppContext":
        store = AlbumStore(config.output_path)
        # Reconcile albums.json with disk in case files were moved by hand.
        store.sync()
        manifest = Manifest(store.root / MANIFEST_FILE, root=store.root)
        pipeline = EditPipeline(config)
        runner = BatchRunner(config, store, pipeline, manifest=manifest)
        thumbnails = ThumbnailCache(store.staging_dir / "thumbs", size=config.thumbnail_size)
        return cls(config=config, store=store, pipeline=pipeline, runner=runner,
                   manifest=manifest, thumbnails=thumbnails)
