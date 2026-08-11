"""Albums tab: overview, nesting, rename, delete, download."""

from __future__ import annotations

import logging

import gradio as gr

from ..storage import UNSORTED, UNSORTED_LABEL, StorageError
from .common import TOP_LEVEL, album_choices, album_label, parent_value
from .context import AppContext

log = logging.getLogger(__name__)


def build_albums_tab(ctx: AppContext) -> dict:
    store = ctx.store

    gr.Markdown("### Albums")

    with gr.Row():
        with gr.Column(scale=3):
            overview = gr.Gallery(
                label="Albums", columns=4, height=380, object_fit="cover",
                allow_preview=False, show_label=False,
            )
            tree_view = gr.Markdown()
        with gr.Column(scale=2):
            album_picker = gr.Dropdown(
                choices=album_choices(store), label="Album", interactive=True,
            )
            album_details = gr.Markdown()

            with gr.Group():
                gr.Markdown("**New album**")
                new_name = gr.Textbox(label="Name", placeholder="Album name")
                new_parent = gr.Dropdown(
                    choices=album_choices(store, include_unsorted=False,
                                          include_top_level=True),
                    value=TOP_LEVEL, label="Nest it under", interactive=True,
                )
                create_button = gr.Button("Create album")

            with gr.Group():
                gr.Markdown("**Rename**")
                rename_input = gr.Textbox(label="New name", placeholder="New album name")
                rename_button = gr.Button("Rename")

            with gr.Group():
                gr.Markdown("**Move**")
                move_parent = gr.Dropdown(
                    choices=album_choices(store, include_unsorted=False,
                                          include_top_level=True),
                    value=TOP_LEVEL, label="Move under", interactive=True,
                )
                move_button = gr.Button("Move album")

            with gr.Group():
                gr.Markdown("**Download**")
                include_nested_zip = gr.Checkbox(
                    value=True, label="Include sub-albums (as nested folders)",
                )
                download_button = gr.Button("Prepare zip")
                download_file = gr.File(label="Zip", interactive=False)

            with gr.Accordion("Delete album", open=False):
                gr.Markdown(
                    "Images move back to **Unsorted** and sub-albums are promoted one "
                    "level up, unless you tick the options below."
                )
                delete_confirm = gr.Checkbox(value=False, label="Yes, delete this album")
                delete_recursive = gr.Checkbox(
                    value=False, label="…and delete its sub-albums too",
                )
                delete_images = gr.Checkbox(
                    value=False, label="…and delete the image files too",
                )
                delete_button = gr.Button("Delete album", variant="stop")

            status = gr.Markdown()

    refresh_button = gr.Button("↻ Refresh albums")

    # -- rendering --------------------------------------------------------
    def covers() -> tuple[list, str]:
        """Cover thumbnails for albums that have images, plus a text tree."""
        items = []
        for album, depth in store.album_tree():
            direct = store.list_images(album.slug)
            nested_total = store.count_images(album.slug, include_descendants=True)
            if direct:
                thumb = ctx.thumbnails.get(store.album_dir(album.slug) / direct[0])
                label = store.path_name(album.slug)
                caption = f"{label} — {len(direct)}"
                if nested_total != len(direct):
                    caption += f" (+{nested_total - len(direct)} nested)"
                items.append((thumb, caption))
            elif nested_total:
                # No image of its own: borrow the first from a sub-album.
                located = store.list_images_located(album.slug, include_descendants=True)
                slug, name = located[0]
                thumb = ctx.thumbnails.get(store.dir_for(slug) / name)
                items.append((thumb, f"{store.path_name(album.slug)} — "
                                     f"0 ({nested_total} nested)"))
        tree = render_tree()
        return items, tree

    def render_tree() -> str:
        rows = store.album_tree()
        if not rows:
            return "_No albums yet — create one above, or from the Gallery tab._"
        lines = ["**Album tree**", ""]
        for album, depth in rows:
            direct = len(store.list_images(album.slug))
            nested = store.count_images(album.slug, include_descendants=True)
            count = f"{direct}" if nested == direct else f"{direct} (+{nested - direct})"
            lines.append(f"{'    ' * depth}- **{album.name}** — {count} image(s)")
        return "\n".join(lines)

    def details(value: str | None) -> str:
        if not value:
            return ""
        if value == UNSORTED:
            return (f"**{UNSORTED_LABEL}** · {len(store.list_unsorted())} image(s)\n\n"
                    "_Unsorted is not a real album: it can be downloaded, but not "
                    "renamed, moved or deleted._")
        try:
            album = store.get_album(value)
        except StorageError:
            return ""
        direct = len(store.list_images(value))
        nested = store.count_images(value, include_descendants=True)
        children = store.children(value)
        lines = [
            f"**{store.path_name(value)}**",
            "",
            f"- {direct} image(s) directly, {nested} including sub-albums",
            f"- {len(children)} sub-album(s)",
            f"- Folder: `{store.album_dir(album.slug)}`",
        ]
        return "\n".join(lines)

    def render(selected: str | None = None, message: str = ""):
        choices = album_choices(store)
        values = [value for _, value in choices]
        if selected not in values:
            selected = values[0] if values else None
        items, tree = covers()
        # An album cannot be moved into itself or its own descendants.
        exclude = set()
        if selected and selected != UNSORTED:
            try:
                exclude = {selected} | {a.slug for a in store.descendants(selected)}
            except StorageError:
                exclude = set()
        parent_choices = album_choices(store, include_unsorted=False,
                                       include_top_level=True)
        move_choices = album_choices(store, include_unsorted=False,
                                     include_top_level=True, exclude=exclude)
        return (
            gr.update(value=items),
            gr.update(value=tree),
            gr.update(choices=choices, value=selected),
            gr.update(value=details(selected)),
            gr.update(choices=parent_choices),
            gr.update(choices=move_choices),
            gr.update(value=message),
        )

    render_outputs = [overview, tree_view, album_picker, album_details, new_parent,
                      move_parent, status]

    # -- operations -------------------------------------------------------
    def create(name, parent):
        if not (name or "").strip():
            return render(None, "⚠️ Enter a name for the new album.")
        try:
            album = store.create_album(name, parent=parent_value(parent))
        except StorageError as exc:
            return render(None, f"❌ {exc}")
        return render(album.slug, f"✅ Created **{store.path_name(album.slug)}**.")

    def rename(value, new_label):
        if not value:
            return render(value, "⚠️ Pick an album first.")
        if value == UNSORTED:
            return render(value, "⚠️ Unsorted cannot be renamed.")
        if not (new_label or "").strip():
            return render(value, "⚠️ Enter a new name.")
        try:
            album = store.rename_album(value, new_label)
        except StorageError as exc:
            return render(value, f"❌ {exc}")
        return render(album.slug, f"✅ Renamed to **{album.name}**.")

    def move(value, new_parent_value):
        if not value:
            return render(value, "⚠️ Pick an album first.")
        if value == UNSORTED:
            return render(value, "⚠️ Unsorted cannot be moved.")
        try:
            album = store.move_album(value, parent_value(new_parent_value))
        except StorageError as exc:
            return render(value, f"❌ {exc}")
        return render(album.slug, f"✅ Moved to **{store.path_name(album.slug)}**.")

    def delete(value, confirm, recursive, also_files):
        if not value:
            return render(value, "⚠️ Pick an album first.")
        if value == UNSORTED:
            return render(value, "⚠️ Unsorted cannot be deleted.")
        if not confirm:
            return render(value, "⚠️ Tick *Yes, delete this album* to confirm.")
        name = album_label(store, value, full_path=True)
        try:
            result = store.delete_album(value, delete_images=also_files,
                                        recursive=recursive)
        except StorageError as exc:
            return render(value, f"❌ {exc}")
        if also_files:
            ctx.manifest.prune_missing()
            ctx.manifest.save_if_dirty()
        parts = [f"{result['images']} image(s) "
                 + ("deleted" if also_files else "moved to Unsorted")]
        if result["albums"] > 1:
            parts.append(f"{result['albums'] - 1} sub-album(s) deleted")
        elif not recursive:
            parts.append("sub-albums promoted")
        return render(None, f"🗑️ Deleted **{name}** ({', '.join(parts)}).")

    def download(value, include_nested):
        if not value:
            return None, "⚠️ Pick an album first."
        try:
            archive = store.zip_album(value, include_descendants=include_nested)
        except StorageError as exc:
            return None, f"❌ {exc}"
        return str(archive), f"✅ Ready: `{archive.name}`"

    # -- wiring -----------------------------------------------------------
    refresh_button.click(lambda value: render(value), inputs=album_picker,
                         outputs=render_outputs)
    album_picker.change(
        # Only the panels that depend on which album is selected; writing back to
        # album_picker from its own change event risks an update loop.
        lambda value: (gr.update(value=details(value)),
                       gr.update(choices=album_choices(
                           store, include_unsorted=False, include_top_level=True,
                           exclude=({value} | {a.slug for a in store.descendants(value)})
                           if value and value != UNSORTED else set()))),
        inputs=album_picker, outputs=[album_details, move_parent])
    create_button.click(create, inputs=[new_name, new_parent], outputs=render_outputs)
    rename_button.click(rename, inputs=[album_picker, rename_input], outputs=render_outputs)
    move_button.click(move, inputs=[album_picker, move_parent], outputs=render_outputs)
    delete_button.click(delete,
                        inputs=[album_picker, delete_confirm, delete_recursive,
                                delete_images],
                        outputs=render_outputs)
    download_button.click(download, inputs=[album_picker, include_nested_zip],
                          outputs=[download_file, status])

    return {
        "render": render,
        "outputs": render_outputs,
        "inputs": [album_picker],
        # Exposed so the handlers can be exercised directly in tests.
        "handlers": {"create": create, "rename": rename, "move": move, "delete": delete,
                     "download": download, "details": details, "covers": covers,
                     "tree": render_tree},
    }
