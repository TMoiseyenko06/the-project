"""Run tab: pick a source, type a prompt, watch it work."""

from __future__ import annotations

import logging

import gradio as gr

from ..batch import BatchError, describe_progress, stage_uploads
from ..prompts import find_preset
from ..storage import UNSORTED, StorageError
from .common import TOP_LEVEL, album_choices, parent_value
from .context import AppContext

log = logging.getLogger(__name__)

FOLDER_MODE = "Folder path"
ZIP_MODE = "Zip upload"
UPLOAD_MODE = "Upload images"
NO_PRESET = "— none —"


def preset_choices(ctx: AppContext) -> list[str]:
    return [NO_PRESET] + [p.name for p in ctx.presets]


def preset_status(ctx: AppContext) -> str:
    if ctx.preset_error:
        return f"⚠️ `prompts.json`: {ctx.preset_error}"
    if not ctx.presets:
        return ("_No presets. Create a `prompts.json` with "
                "`[{\"name\": ..., \"prompt\": ..., \"tags\": {...}}]` to add some._")
    return f"_{len(ctx.presets)} preset(s) loaded._"


def parse_manual_tags(text: str) -> dict[str, list[str]]:
    """Parse ``"pose=Sitting, style=b&w"`` into ``{"pose": ["Sitting"], ...}``.

    Repeating a category accumulates values rather than overwriting, so
    ``person=Alex, person=Jordan`` tags both.
    """
    tags: dict[str, list[str]] = {}
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"tag {chunk!r} must be in category=value form (e.g. pose=Sitting)")
        category, _, value = chunk.partition("=")
        category, value = category.strip().lower(), value.strip()
        if not category or not value:
            raise ValueError(
                f"tag {chunk!r} must be in category=value form (e.g. pose=Sitting)")
        tags.setdefault(category, [])
        if value not in tags[category]:
            tags[category].append(value)
    return tags


def merge_tags(*sources: dict[str, list[str]]) -> dict[str, list[str]]:
    """Combine tag dicts, unioning values per category (later never clobbers earlier)."""
    merged: dict[str, list[str]] = {}
    for source in sources:
        for category, values in (source or {}).items():
            merged.setdefault(category, [])
            for value in values:
                if value not in merged[category]:
                    merged[category].append(value)
    return merged


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
                choices=[FOLDER_MODE, ZIP_MODE, UPLOAD_MODE], value=FOLDER_MODE,
                label="Image source",
            )
            folder_input = gr.Textbox(
                label="Folder path",
                placeholder="/workspace/input_images",
                info="A folder on this machine. Subfolders are included by default.",
            )
            zip_input = gr.File(
                label="Zip file", file_types=[".zip"], type="filepath", visible=False,
            )
            images_input = gr.File(
                label="Images", file_count="multiple", file_types=["image"],
                type="filepath", visible=False,
                # On a phone this opens the native photo picker, which supports
                # multi-select straight from the camera roll or camera.
            )
            with gr.Group():
                gr.Markdown("**Preset**")
                with gr.Row():
                    preset_picker = gr.Dropdown(
                        choices=preset_choices(ctx), value=NO_PRESET, label="Prompt preset",
                        interactive=True, scale=3,
                        info="Fills in the prompt and applies its tags to every result.",
                    )
                    reload_presets_button = gr.Button("↻", scale=1,
                                                      min_width=48)
                preset_note = gr.Markdown(preset_status(ctx))
            prompt_input = gr.Textbox(
                label="Prompt", lines=3,
                placeholder="e.g. make it look like a watercolour painting",
                info="Applied to every image in the batch.",
            )
            with gr.Group():
                gr.Markdown("**Tag these results**")
                manual_tags = gr.Textbox(
                    label="Tags", placeholder="pose=Sitting, style=b&w",
                    info="category=value pairs, comma separated. Applied to every "
                         "image in this batch, on top of any preset tags.",
                )
                detect_faces = gr.Checkbox(
                    value=False, label="Detect faces and auto-tag people",
                    info="Groups the same face across runs under a stable id you "
                         "can name later in the Tags tab. Needs `insightface`.",
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
                gr.update(visible=mode == ZIP_MODE),
                gr.update(visible=mode == UPLOAD_MODE))

    source_mode.change(toggle_source, inputs=source_mode,
                       outputs=[folder_input, zip_input, images_input])

    def apply_preset(name):
        """Fill the prompt from the chosen preset (its tags apply at run time)."""
        preset = find_preset(ctx.presets, name or "")
        if preset is None:
            return gr.update(), gr.update(value="")
        summary = ", ".join(f"`{c}={v}`" for c, values in preset.tags.items()
                            for v in values)
        return (gr.update(value=preset.prompt),
                gr.update(value=f"Tags from preset: {summary}" if summary
                          else "_This preset has no tags._"))

    def reload_presets():
        ctx.reload_presets()
        return (gr.update(choices=preset_choices(ctx), value=NO_PRESET),
                gr.update(value=preset_status(ctx)))

    preset_picker.change(apply_preset, inputs=preset_picker,
                         outputs=[prompt_input, preset_note])
    reload_presets_button.click(reload_presets, outputs=[preset_picker, preset_note])

    def run_batch(mode, folder, zip_path, images, prompt, target, new_name, new_parent,
                  resume, recursive, preset_name, manual_tag_text, want_faces):
        if mode == UPLOAD_MODE:
            if not images:
                return "❌ Choose one or more images first.", ""
            try:
                source = stage_uploads(images, ctx.store.staging_dir)
            except BatchError as exc:
                return f"❌ {exc}", ""
        else:
            source = folder if mode == FOLDER_MODE else zip_path
        if not source:
            return "❌ Choose a folder or upload a zip first.", ""
        if not (prompt or "").strip():
            return "❌ Enter a prompt.", ""

        # A name typed here wins over the dropdown: it is the more deliberate act.
        if (new_name or "").strip():
            try:
                target = ctx.store.create_album(new_name, parent=parent_value(new_parent)).slug
            except StorageError as exc:
                return f"❌ {exc}", ""

        try:
            manual = parse_manual_tags(manual_tag_text)
        except ValueError as exc:
            return f"❌ {exc}", ""
        preset = find_preset(ctx.presets, preset_name or "")
        batch_tags = merge_tags(preset.tags if preset else {}, manual)

        # Start on a background thread and return immediately. The run is not
        # tied to this browser connection, so refreshing or closing the page
        # leaves it going — results keep landing in the output directory.
        try:
            ctx.background.start(source=source, prompt=prompt, resume=resume,
                                 recursive=recursive, target_album=target,
                                 tags=batch_tags, detect_faces=want_faces)
        except BatchError as exc:
            return f"❌ {exc}", ""
        except Exception as exc:  # noqa: BLE001 - never leave the UI without a reason
            log.exception("Could not start batch")
            return f"❌ Unexpected error: {type(exc).__name__}: {exc}", ""
        return ("🚀 Started. This keeps running if you close or refresh the page — "
                "results appear in the Gallery as they finish."), ""

    run_event = run_button.click(
        run_batch,
        inputs=[source_mode, folder_input, zip_input, images_input, prompt_input,
                target_album, new_album_name, new_album_parent, resume_input,
                recursive_input, preset_picker, manual_tags, detect_faces],
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
        if not ctx.background.is_active:
            return "Nothing is running."
        ctx.background.cancel()
        return "🛑 Stopping after the current image…"

    stop_button.click(request_stop, outputs=status)

    def poll_status():
        """Render the background run's latest state.

        Reads from the runner rather than driving it, so this reconnects
        cleanly to a run already in flight after a refresh.
        """
        snap = ctx.background.snapshot()
        if snap["error"]:
            return f"❌ {snap['error']}", ""
        update = snap["progress"]
        if update is None:
            if snap["active"]:
                return "🚀 Starting…", ""
            return ("Ready." if not snap["ever_ran"] else "Finished."), ""
        text = describe_progress(update)
        if snap["active"]:
            text += "\n\n_Running in the background — safe to close this page._"
        return text, (update.message if update.finished else "")

    # Polls whether or not this browser started the run, so a refreshed page
    # picks the status back up instead of showing a blank slate.
    status_timer = gr.Timer(2.0)
    status_timer.tick(poll_status, outputs=[status, summary])

    return {
        "run_event": run_event,
        "status": status,
        "refresh": refresh_albums,
        "refresh_inputs": [target_album],
        "refresh_outputs": [target_album, new_album_parent, new_album_name],
        "handlers": {"run_batch": run_batch, "refresh_albums": refresh_albums,
                     "poll_status": poll_status, "request_stop": request_stop},
    }
