"""Assembles the three tabs into a Gradio app and launches it."""

from __future__ import annotations

import logging

import gradio as gr

from ..config import Config
from .context import AppContext
from .gallery_tab import build_gallery_tab
from .run_tab import build_run_tab
from .tags_tab import build_tags_tab

log = logging.getLogger(__name__)

CSS = """
.gradio-container { max-width: 1500px !important; }
footer { display: none !important; }
"""


def build_ui(config: Config) -> tuple[gr.Blocks, AppContext]:
    ctx = AppContext.create(config)

    with gr.Blocks(title="Batch Image Editor", css=CSS, theme=gr.themes.Soft(),
                   analytics_enabled=False) as demo:
        gr.Markdown("# Batch Image Editor")

        with gr.Tabs():
            with gr.TabItem("Run") as run_tab_item:
                run = build_run_tab(ctx)
            with gr.TabItem("Gallery") as gallery_tab_item:
                gallery = build_gallery_tab(ctx)
            with gr.TabItem("Tags") as tags_tab_item:
                tags = build_tags_tab(ctx)

        # Switching to a tab re-reads from disk, so results from a run that just
        # finished (or edits made in another tab) show up without a manual refresh.
        run_tab_item.select(run["refresh"], inputs=run["refresh_inputs"],
                            outputs=run["refresh_outputs"])
        gallery_tab_item.select(gallery["refresh"], inputs=gallery["inputs"],
                                outputs=gallery["outputs"])
        tags_tab_item.select(tags["refresh"], inputs=tags["inputs"],
                             outputs=tags["outputs"])
        demo.load(gallery["refresh"], inputs=gallery["inputs"], outputs=gallery["outputs"])

    return demo, ctx


def launch(config: Config, preload_model: bool = False) -> None:
    demo, ctx = build_ui(config)

    if preload_model:
        log.info("Preloading model before starting the server…")
        ctx.pipeline.load()

    log.info("Serving on http://%s:%d (outputs in %s)",
             config.host, config.port, ctx.store.root)
    demo.queue(default_concurrency_limit=4).launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        show_error=True,
        # Gradio only serves files from paths it has been told about.
        allowed_paths=[str(ctx.store.root)],
        inbrowser=False,
    )
