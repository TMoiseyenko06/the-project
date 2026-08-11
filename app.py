#!/usr/bin/env python3
"""Entry point: load config, build the UI, serve it.

    python app.py                        # use config.yaml (or defaults)
    python app.py --config other.yaml    # explicit config file
    python app.py --model <hf-id> --port 8000
    python app.py --cli /data/images "make it a pencil sketch"   # headless batch
"""

from __future__ import annotations

import argparse
import logging
import sys

from imagebatch.config import configure_logging, load_config
from imagebatch.storage import AlbumStore

log = logging.getLogger("imagebatch")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image editing tool")
    parser.add_argument("--config", help="path to a YAML config file (default: config.yaml)")
    parser.add_argument("--model", help="override model_id")
    parser.add_argument("--pipeline-class", help="override the diffusers pipeline class")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"],
                        help="override the compute dtype")
    parser.add_argument("--output-dir", help="override the output directory")
    parser.add_argument("--host", help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port (default 7860)")
    parser.add_argument("--share", action="store_true",
                        help="create a public gradio.live tunnel")
    parser.add_argument("--preload", action="store_true",
                        help="load the model at startup instead of on first run")
    parser.add_argument("--cli", nargs=2, metavar=("SOURCE", "PROMPT"),
                        help="run one batch headlessly and exit (no web UI)")
    parser.add_argument("--no-resume", action="store_true",
                        help="with --cli: reprocess images even if already done")
    parser.add_argument("--album", metavar="NAME",
                        help="with --cli: file this batch's results into this album "
                             "(created if needed) instead of unsorted/")
    parser.add_argument("--album-parent", metavar="NAME",
                        help="with --cli and --album: nest the album under this one")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "model_id": args.model,
        "pipeline_class": args.pipeline_class,
        "dtype": args.dtype,
        "output_dir": args.output_dir,
        "host": args.host,
        "port": args.port,
    }
    overrides = {k: v for k, v in mapping.items() if v is not None}
    if args.share:
        overrides["share"] = True
    return overrides


def resolve_cli_album(store: AlbumStore, name: str | None,
                      parent_name: str | None) -> str | None:
    """Find or create the album named on the command line, returning its slug."""
    if not name:
        return None
    parent_slug = None
    if parent_name:
        parent = store.find_by_name(parent_name)
        if parent is None:
            parent = store.create_album(parent_name)
        parent_slug = parent.slug
    existing = store.find_by_name(name, parent=parent_slug)
    if existing is not None:
        return existing.slug
    return store.create_album(name, parent=parent_slug).slug


def run_cli(config, source: str, prompt: str, resume: bool,
            album: str | None = None, album_parent: str | None = None) -> int:
    """Headless batch run — handy over SSH and for smoke-testing a new model."""
    from imagebatch.batch import BatchError, BatchRunner
    from imagebatch.pipeline import EditPipeline
    from imagebatch.storage import StorageError

    store = AlbumStore(config.output_path)
    try:
        target = resolve_cli_album(store, album, album_parent)
    except StorageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    runner = BatchRunner(config, store, EditPipeline(config))
    last_line = ""
    try:
        for progress in runner.run(source, prompt, resume=resume, target_album=target):
            if progress.finished:
                print("\n" + progress.message)
                return 1 if progress.failed else 0
            if progress.total:
                line = (f"\r{progress.done}/{progress.total} "
                        f"(✅ {progress.succeeded} ❌ {progress.failed}) {progress.current[:40]}")
                if line != last_line:
                    sys.stdout.write(line.ljust(90))
                    sys.stdout.flush()
                    last_line = line
    except BatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides=build_overrides(args))
    configure_logging(config.log_level)

    log.info("Model: %s (%s, %s)", config.model_id, config.pipeline_class, config.dtype)
    if config.is_mock:
        log.warning(
            "model_id is 'mock' — the app runs end to end but produces placeholder "
            "images. Set model_id in config.yaml or IMGBATCH_MODEL_ID for real edits."
        )

    if args.cli:
        source, prompt = args.cli
        return run_cli(config, source, prompt, resume=not args.no_resume,
                       album=args.album, album_parent=args.album_parent)

    from imagebatch.ui import launch

    launch(config, preload_model=args.preload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
