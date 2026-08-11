"""Gallery tab: browse results, multi-select, and file them into albums."""

from __future__ import annotations

import logging

import gradio as gr

from ..storage import UNSORTED, StorageError
from .common import album_choices, album_label, full_path, gallery_items, page_count, page_slice
from .context import AppContext

log = logging.getLogger(__name__)


def build_gallery_tab(ctx: AppContext) -> dict:
    store = ctx.store
    page_size = ctx.config.gallery_page_size

    # -- state ------------------------------------------------------------
    names_state = gr.State([])      # every filename in the current album
    selected_state = gr.State([])   # filenames the user has picked
    page_state = gr.State(1)

    gr.Markdown(
        "### Gallery\n"
        "Click images to add them to the selection (the pane on the right shows the "
        "full-size image). Then assign the selection to an album."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                album_filter = gr.Dropdown(
                    choices=album_choices(store), value=UNSORTED, label="Album",
                    interactive=True, scale=3,
                )
                refresh_button = gr.Button("↻ Refresh", scale=1)
            gallery = gr.Gallery(
                label="Images", columns=5, height=560, object_fit="cover",
                allow_preview=False,  # click selects; the right pane does enlargement
                show_label=False,
            )
            with gr.Row():
                prev_button = gr.Button("← Previous")
                page_info = gr.Markdown("Page 1 / 1")
                next_button = gr.Button("Next →")

        with gr.Column(scale=2):
            preview = gr.Image(label="Preview", type="filepath", height=320)
            preview_caption = gr.Markdown()
            with gr.Row():
                select_page_button = gr.Button("Select page", size="sm")
                select_all_button = gr.Button("Select all", size="sm")
                clear_button = gr.Button("Clear", size="sm")
            selection_group = gr.CheckboxGroup(
                choices=[], value=[], label="Selection", interactive=True,
            )
            gr.Markdown("**Move selection to an album**")
            target_album = gr.Dropdown(
                choices=album_choices(store), label="Existing album", interactive=True,
            )
            assign_button = gr.Button("Move to album", variant="primary")
            with gr.Row():
                new_album_name = gr.Textbox(
                    label="…or create a new album", placeholder="Album name", scale=2,
                )
                create_button = gr.Button("Create & move", scale=1)
            with gr.Accordion("Danger zone", open=False):
                delete_confirm = gr.Checkbox(
                    value=False, label="Yes, permanently delete the selected images",
                )
                delete_button = gr.Button("Delete selection", variant="stop")
            status = gr.Markdown()

    # -- rendering --------------------------------------------------------
    def render(album: str, page: int, selected: list[str], message: str = "",
               update_filter: bool = True):
        """Build the full set of component updates for the current view.

        ``update_filter`` must be False for handlers bound to
        ``album_filter.change`` — writing back to the component that triggered
        the event risks an update loop.
        """
        album = album or UNSORTED
        names = store.list_images(album)
        selected = [n for n in (selected or []) if n in names]  # drop stale entries
        pages = page_count(len(names), page_size)
        page = min(max(1, int(page or 1)), pages)
        visible = page_slice(names, page, page_size)
        items = gallery_items(store, album, visible, ctx.thumbnails, set(selected))
        info = (f"Page **{page}** / {pages} · {len(names)} image(s) · "
                f"**{len(selected)}** selected")
        return (
            gr.update(value=items),
            gr.update(value=info),
            names,
            selected,
            page,
            gr.update(choices=selected, value=selected),
            gr.update(choices=album_choices(store), value=album) if update_filter
            else gr.update(),
            gr.update(choices=album_choices(store, include_unsorted=True)),
            gr.update(value=message),
        )

    # Every handler returns this same bundle, so wiring stays uniform.
    render_outputs = [gallery, page_info, names_state, selected_state, page_state,
                      selection_group, album_filter, target_album, status]

    def refresh(album, page, selected):
        return render(album, page, selected)

    def change_album(album):
        # A different album means a fresh selection. The filter itself is left
        # untouched: it is the component that fired this event.
        return render(album, 1, [], update_filter=False)

    def change_page(album, page, selected, delta):
        return render(album, int(page or 1) + delta, selected)

    # -- selection --------------------------------------------------------
    def on_select(album, page, names, selected, evt: gr.SelectData):
        album = album or UNSORTED
        visible = page_slice(list(names or []), int(page or 1), page_size)
        if evt.index is None or evt.index >= len(visible):
            return gr.update(), selected, gr.update(), gr.update(), gr.update()
        name = visible[evt.index]
        selected = list(selected or [])
        if name in selected:
            selected.remove(name)
        else:
            selected.append(name)
        path = full_path(store, album, name)
        return (
            gr.update(choices=selected, value=selected),
            selected,
            gr.update(value=path),
            gr.update(value=f"`{name}`"),
            gr.update(value=f"{len(selected)} selected"),
        )

    def on_checkbox_input(values):
        # Fires only on real user interaction, so this cannot loop with the
        # programmatic updates made by on_select.
        values = list(values or [])
        return gr.update(choices=values, value=values), values, f"{len(values)} selected"

    def select_visible(album, page, names, selected):
        visible = page_slice(list(names or []), int(page or 1), page_size)
        merged = list(dict.fromkeys(list(selected or []) + visible))
        return render(album, page, merged, f"Selected {len(visible)} on this page.")

    def select_all(album, page, names):
        names = list(names or [])
        return render(album, page, names, f"Selected all {len(names)} image(s).")

    def clear_selection(album, page):
        return render(album, page, [], "Selection cleared.")

    # -- album operations -------------------------------------------------
    def assign(album, target, selected, page):
        selected = list(selected or [])
        if not selected:
            return render(album, page, selected, "⚠️ Nothing selected.")
        if not target:
            return render(album, page, selected, "⚠️ Pick a destination album.")
        if target == album:
            return render(album, page, selected, "⚠️ Those images are already there.")
        try:
            moved = store.assign(selected, album, target)
        except StorageError as exc:
            return render(album, page, selected, f"❌ {exc}")
        message = (f"✅ Moved **{len(moved)}** image(s) to "
                   f"**{album_label(store, target)}**.")
        return render(album, page, [], message)

    def create_and_assign(album, name, selected, page):
        selected = list(selected or [])
        if not (name or "").strip():
            return render(album, page, selected, "⚠️ Enter a name for the new album.")
        try:
            created = store.create_album(name)
        except StorageError as exc:
            return render(album, page, selected, f"❌ {exc}")
        if not selected:
            return render(album, page, selected,
                          f"✅ Created **{created.name}** (nothing selected to move).")
        moved = store.assign(selected, album, created.slug)
        return render(album, page, [],
                      f"✅ Created **{created.name}** and moved {len(moved)} image(s).")

    def delete_selection(album, selected, page, confirm):
        selected = list(selected or [])
        if not selected:
            return render(album, page, selected, "⚠️ Nothing selected.")
        if not confirm:
            return render(album, page, selected,
                          "⚠️ Tick the confirmation box to delete these images.")
        removed = store.delete_images(selected, album)
        ctx.manifest.prune_missing()
        ctx.manifest.save_if_dirty()
        return render(album, page, [], f"🗑️ Deleted **{removed}** image(s).")

    # -- wiring -----------------------------------------------------------
    refresh_button.click(refresh, inputs=[album_filter, page_state, selected_state],
                         outputs=render_outputs)
    album_filter.change(change_album, inputs=album_filter, outputs=render_outputs)
    prev_button.click(lambda a, p, s: change_page(a, p, s, -1),
                      inputs=[album_filter, page_state, selected_state],
                      outputs=render_outputs)
    next_button.click(lambda a, p, s: change_page(a, p, s, 1),
                      inputs=[album_filter, page_state, selected_state],
                      outputs=render_outputs)

    gallery.select(
        on_select,
        inputs=[album_filter, page_state, names_state, selected_state],
        outputs=[selection_group, selected_state, preview, preview_caption, status],
    )
    selection_group.input(on_checkbox_input, inputs=selection_group,
                          outputs=[selection_group, selected_state, status])

    select_page_button.click(select_visible,
                             inputs=[album_filter, page_state, names_state, selected_state],
                             outputs=render_outputs)
    select_all_button.click(select_all, inputs=[album_filter, page_state, names_state],
                            outputs=render_outputs)
    clear_button.click(clear_selection, inputs=[album_filter, page_state],
                       outputs=render_outputs)

    assign_button.click(assign,
                        inputs=[album_filter, target_album, selected_state, page_state],
                        outputs=render_outputs)
    create_button.click(create_and_assign,
                        inputs=[album_filter, new_album_name, selected_state, page_state],
                        outputs=render_outputs)
    delete_button.click(delete_selection,
                        inputs=[album_filter, selected_state, page_state, delete_confirm],
                        outputs=render_outputs)

    return {
        "refresh": refresh,
        "outputs": render_outputs,
        "inputs": [album_filter, page_state, selected_state],
        # Exposed so the handlers can be exercised directly in tests.
        "handlers": {
            "render": render,
            "change_album": change_album,
            "select_visible": select_visible,
            "select_all": select_all,
            "clear_selection": clear_selection,
            "assign": assign,
            "create_and_assign": create_and_assign,
            "delete_selection": delete_selection,
        },
    }
