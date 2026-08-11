"""Run tab: pick a source, type a prompt, watch it work."""

from __future__ import annotations

import logging

import gradio as gr

from ..batch import BatchError, describe_progress
from ..storage import UNSORTED, StorageError
from .common import TOP_LEVEL, album_choices, parent_value
from .context import AppContext

log = logging.getLogger(__name__)

FOLDER_MODE = "Folder path"
ZIP_MODE = "Zip upload"


def build_run_tab(ctx: AppContext) -> dict:
    cfg = ctx.config

    gr.Markdown(
        f"### Batch edit\n"
        f"Model: `{cfg.model_id}` · pipeline: `{cfg.pipeline_class}` · "
        f"dtype: `{cfg.dtype}` · batch size: `{cfg.batch_size}`"
        + ("\n\n⚠️ **`model_id` is `mock`** — outputs are placeholder transforms, not real "
           "edits. Set a real model in `config.yaml` or `IMGBATCH_MODEL_ID`."
           if cfg.is_mock else "")
    )

    with gr.Row():
        with gr.Column(scale=1):
            source_mode = gr.Radio(
                choices=[FOLDER_MODE, ZIP_MODE], value=FOLDER_MODE, label="Image source",
            )
            folder_input = gr.Textbox(
                label="Folder path",
                placeholder="/workspace/input_images",
                info="A folder on this machine. Subfolders are included by default.",
            )
            zip_input = gr.File(
                label="Zip file", file_types=[".zip"], type="filepath", visible=False,
            )
            prompt_input = gr.Textbox(
                label="Prompt", lines=3,
                placeholder="e.g. make it look like a watercolour painting",
                info="Applied to every image in the batch.",
            )
            with gr.Group():
                gr.Markdown("**Send results to**")
                target_album = gr.Dropdown(
                    choices=album_choices(ctx.store), value=UNSORTED, label="Album",
                    interactive=True,
                    info="Every image in this batch is filed here as it is produced.",
                )
                with gr.Row():
                    new_album_name = gr.Textbox(
                        label="…or create a new album for this batch",
                        placeholder="Leave blank to use the album above", scale=2,
                    )
                    new_album_parent = gr.Dropdown(
                        choices=album_choices(ctx.store, include_unsorted=False,
                                              include_top_level=True),
                        value=TOP_LEVEL, label="Nest it under", scale=1,
                    )
            with gr.Accordion("Options", open=False):
                resume_input = gr.Checkbox(
                    value=True, label="Resume (skip images already done with this prompt)",
                )
                recursive_input = gr.Checkbox(value=True, label="Include subfolders")
                gr.Markdown(
                    "Generation settings (steps, guidance, seed, dtype) come from "
                    "`config.yaml`. Settings the loaded model doesn't accept are "
                    "dropped automatically."
                )
            with gr.Row():
                run_button = gr.Button("Run batch", variant="primary", scale=3)
                stop_button = gr.Button("Stop", variant="stop", scale=1)

        with gr.Column(scale=1):
            status = gr.Markdown("Ready.")
            summary = gr.Markdown()

    def toggle_source(mode: str):
        return (gr.update(visible=mode == FOLDER_MODE),
                gr.update(visible=mode == ZIP_MODE))

    source_mode.change(toggle_source, inputs=source_mode, outputs=[folder_input, zip_input])

    def run_batch(mode, folder, zip_path, prompt, target, new_name, new_parent,
                  resume, recursive, progress=gr.Progress()):
        source = folder if mode == FOLDER_MODE else zip_path
        if not source:
            yield "❌ Choose a folder or upload a zip first.", ""
            return
        if not (prompt or "").strip():
            yield "❌ Enter a prompt.", ""
            return

        # A name typed here wins over the dropdown: it is the more deliberate act.
        if (new_name or "").strip():
            try:
                target = ctx.store.create_album(new_name, parent=parent_value(new_parent)).slug
            except StorageError as exc:
                yield f"❌ {exc}", ""
                return

        yield "Starting…", ""
        try:
            for update in ctx.runner.run(source, prompt, resume=resume,
                                         recursive=recursive, target_album=target):
                if update.total:
                    progress(update.fraction, desc=f"{update.done}/{update.total}")
                if update.finished:
                    yield describe_progress(update), update.message
                else:
                    yield describe_progress(update), ""
        except BatchError as exc:
            yield f"❌ {exc}", ""
        except Exception as exc:  # noqa: BLE001 - never leave the UI without a reason
            log.exception("Batch run failed")
            yield f"❌ Unexpected error: {type(exc).__name__}: {exc}", ""

    run_event = run_button.click(
        run_batch,
        inputs=[source_mode, folder_input, zip_input, prompt_input, target_album,
                new_album_name, new_album_parent, resume_input, recursive_input],
        outputs=[status, summary],
        concurrency_limit=1,  # one GPU, one batch at a time
    )

    def refresh_albums(current):
        """Re-read albums so the destination list reflects the other tabs."""
        choices = album_choices(ctx.store)
        values = [value for _, value in choices]
        return (
            gr.update(choices=choices, value=current if current in values else UNSORTED),
            gr.update(choices=album_choices(ctx.store, include_unsorted=False,
                                            include_top_level=True)),
            gr.update(value=""),
        )

    def request_stop():
        if not ctx.runner.is_running:
            return "Nothing is running."
        ctx.runner.cancel()
        return "🛑 Stopping after the current image…"

    stop_button.click(request_stop, outputs=status)

    return {
        "run_event": run_event,
        "status": status,
        "refresh": refresh_albums,
        "refresh_inputs": [target_album],
        "refresh_outputs": [target_album, new_album_parent, new_album_name],
        "handlers": {"run_batch": run_batch, "refresh_albums": refresh_albums},
    }
