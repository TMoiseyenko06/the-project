"""Albums tab: overview, rename, delete, download."""

from __future__ import annotations

import logging

import gradio as gr

from ..storage import UNSORTED, UNSORTED_LABEL, StorageError
from .common import album_choices, album_label
from .context import AppContext

log = logging.getLogger(__name__)


def build_albums_tab(ctx: AppContext) -> dict:
    store = ctx.store

    gr.Markdown("### Albums")

    with gr.Row():
        with gr.Column(scale=3):
            overview = gr.Gallery(
                label="Albums", columns=4, height=420, object_fit="cover",
                allow_preview=False, show_label=False,
            )
            overview_empty = gr.Markdown()
        with gr.Column(scale=2):
            album_picker = gr.Dropdown(
                choices=album_choices(store), label="Album", interactive=True,
            )
            album_details = gr.Markdown()

            with gr.Group():
                gr.Markdown("**Rename**")
                rename_input = gr.Textbox(label="New name", placeholder="New album name")
                rename_button = gr.Button("Rename")

            with gr.Group():
                gr.Markdown("**Download**")
                download_button = gr.Button("Prepare zip")
                download_file = gr.File(label="Zip", interactive=False)

            with gr.Accordion("Delete album", open=False):
                gr.Markdown(
                    "Deleting an album moves its images back to **Unsorted** unless you "
                    "also tick *delete the image files*."
                )
                delete_confirm = gr.Checkbox(value=False, label="Yes, delete this album")
                delete_images = gr.Checkbox(value=False, label="…and delete the image files too")
                delete_button = gr.Button("Delete album", variant="stop")

            status = gr.Markdown()

    refresh_button = gr.Button("↻ Refresh albums")

    # -- rendering --------------------------------------------------------
    def covers() -> tuple[list, str]:
        items = []
        for album in store.list_albums():
            images = store.list_images(album.slug)
            if images:
                thumb = ctx.thumbnails.get(store.album_dir(album.slug) / images[0])
                items.append((thumb, f"{album.name} — {len(images)} image(s)"))
            else:
                items.append((None, f"{album.name} — empty"))
        # Gradio's gallery cannot render a None image, so empty albums are
        # described in the note below the grid instead.
        renderable = [item for item in items if item[0]]
        empty = [item[1] for item in items if not item[0]]
        note = ""
        if not items:
            note = "_No albums yet — create one from the Gallery tab._"
        elif empty:
            note = "_Empty albums: " + ", ".join(empty) + "_"
        return renderable, note

    def details(value: str | None) -> str:
        if not value:
            return ""
        count = len(store.list_images(value))
        if value == UNSORTED:
            return (f"**{UNSORTED_LABEL}** · {count} image(s)\n\n"
                    "_Unsorted is not a real album: it can be downloaded, but not "
                    "renamed or deleted._")
        try:
            album = store.get_album(value)
        except StorageError:
            return ""
        return (f"**{album.name}** · {count} image(s)\n\n"
                f"Folder: `{store.album_dir(album.slug)}`")

    def render(selected: str | None = None, message: str = ""):
        choices = album_choices(store)
        values = [value for _, value in choices]
        if selected not in values:
            selected = values[0] if values else None
        items, note = covers()
        return (
            gr.update(value=items),
            gr.update(value=note),
            gr.update(choices=choices, value=selected),
            gr.update(value=details(selected)),
            gr.update(value=message),
        )

    render_outputs = [overview, overview_empty, album_picker, album_details, status]

    # -- operations -------------------------------------------------------
    def rename(value, new_name):
        if not value:
            return render(value, "⚠️ Pick an album first.")
        if value == UNSORTED:
            return render(value, "⚠️ Unsorted cannot be renamed.")
        if not (new_name or "").strip():
            return render(value, "⚠️ Enter a new name.")
        try:
            album = store.rename_album(value, new_name)
        except StorageError as exc:
            return render(value, f"❌ {exc}")
        return render(album.slug, f"✅ Renamed to **{album.name}**.")

    def delete(value, confirm, also_files):
        if not value:
            return render(value, "⚠️ Pick an album first.")
        if value == UNSORTED:
            return render(value, "⚠️ Unsorted cannot be deleted.")
        if not confirm:
            return render(value, "⚠️ Tick *Yes, delete this album* to confirm.")
        name = album_label(store, value)
        try:
            affected = store.delete_album(value, delete_images=also_files)
        except StorageError as exc:
            return render(value, f"❌ {exc}")
        if also_files:
            ctx.manifest.prune_missing()
            ctx.manifest.save_if_dirty()
            note = f"{affected} image(s) deleted"
        else:
            note = f"{affected} image(s) moved to Unsorted"
        return render(None, f"🗑️ Deleted album **{name}** ({note}).")

    def download(value):
        if not value:
            return None, "⚠️ Pick an album first."
        try:
            archive = store.zip_album(value)
        except StorageError as exc:
            return None, f"❌ {exc}"
        return str(archive), f"✅ Ready: `{archive.name}`"

    # -- wiring -----------------------------------------------------------
    refresh_button.click(lambda value: render(value), inputs=album_picker,
                         outputs=render_outputs)
    album_picker.change(lambda value: gr.update(value=details(value)),
                        inputs=album_picker, outputs=album_details)
    rename_button.click(rename, inputs=[album_picker, rename_input], outputs=render_outputs)
    delete_button.click(delete, inputs=[album_picker, delete_confirm, delete_images],
                        outputs=render_outputs)
    download_button.click(download, inputs=album_picker, outputs=[download_file, status])

    return {
        "render": render,
        "outputs": render_outputs,
        "inputs": [album_picker],
        # Exposed so the handlers can be exercised directly in tests.
        "handlers": {"rename": rename, "delete": delete, "download": download,
                     "details": details, "covers": covers},
    }
