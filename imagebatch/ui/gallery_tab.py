"""Gallery tab: browse results, multi-select, and file them into albums."""

from __future__ import annotations

import logging
from collections import defaultdict

import gradio as gr

from ..storage import UNSORTED, StorageError
from .common import (TOP_LEVEL, album_choices, album_label, page_count, page_slice,
                     parent_value)
from .context import AppContext

log = logging.getLogger(__name__)


def token_for(album: str, name: str) -> str:
    """Identify an image as ``album/filename``.

    A bare filename is not unique once the view spans sub-albums — two albums
    can each hold ``img_01.png`` — so selections carry their album with them.
    """
    return f"{album}/{name}"


def split_token(token: str) -> tuple[str, str]:
    album, _, name = token.partition("/")
    return album, name


def build_gallery_tab(ctx: AppContext) -> dict:
    store = ctx.store
    page_size = ctx.config.gallery_page_size

    # -- state ------------------------------------------------------------
    tokens_state = gr.State([])    # every "album/filename" in the current view
    selected_state = gr.State([])  # the tokens the user has picked
    page_state = gr.State(1)

    gr.Markdown(
        "### Gallery\n"
        "Click images to add them to the selection (the pane on the right shows the "
        "full-size image). Then move the selection into an album."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                album_filter = gr.Dropdown(
                    choices=album_choices(store), value=UNSORTED, label="Album",
                    interactive=True, scale=3,
                )
                refresh_button = gr.Button("↻ Refresh", scale=1)
            include_nested = gr.Checkbox(
                value=False, label="Include sub-albums",
                info="Show images from albums nested inside this one.",
            )
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
            with gr.Group():
                gr.Markdown("**…or create a new album**")
                new_album_name = gr.Textbox(label="Name", placeholder="Album name")
                new_album_parent = gr.Dropdown(
                    choices=album_choices(store, include_unsorted=False,
                                          include_top_level=True),
                    value=TOP_LEVEL, label="Nest it under", interactive=True,
                )
                create_button = gr.Button("Create & move")
            with gr.Accordion("Danger zone", open=False):
                delete_confirm = gr.Checkbox(
                    value=False, label="Yes, permanently delete the selected images",
                )
                delete_button = gr.Button("Delete selection", variant="stop")
            status = gr.Markdown()

    # -- rendering --------------------------------------------------------
    def view_tokens(album: str, nested: bool) -> list[str]:
        located = store.list_images_located(album, include_descendants=nested)
        return [token_for(slug, name) for slug, name in located]

    def caption_for(token: str, album: str, nested: bool) -> str:
        slug, name = split_token(token)
        if nested and slug != album:
            return f"{album_label(store, slug)} / {name}"
        return name

    def render(album: str, page: int, selected: list[str], message: str = "",
               nested: bool = False, update_filter: bool = True):
        """Build the full set of component updates for the current view.

        ``update_filter`` must be False for handlers bound to
        ``album_filter.change`` — writing back to the component that triggered
        the event risks an update loop.
        """
        album = album or UNSORTED
        try:
            tokens = view_tokens(album, nested)
        except StorageError:
            # The album was deleted in the other tab; fall back to Unsorted.
            album, tokens = UNSORTED, view_tokens(UNSORTED, False)
        selected = [t for t in (selected or []) if t in tokens]  # drop stale entries
        pages = page_count(len(tokens), page_size)
        page = min(max(1, int(page or 1)), pages)
        visible = page_slice(tokens, page, page_size)

        items = []
        chosen = set(selected)
        for token in visible:
            slug, name = split_token(token)
            path = store.dir_for(slug) / name
            if not path.is_file():
                continue
            caption = caption_for(token, album, nested)
            items.append((ctx.thumbnails.get(path), f"✅ {caption}" if token in chosen
                          else caption))

        info = (f"Page **{page}** / {pages} · {len(tokens)} image(s) · "
                f"**{len(selected)}** selected")
        selection_choices = [(caption_for(t, album, nested), t) for t in selected]
        return (
            gr.update(value=items),
            gr.update(value=info),
            tokens,
            selected,
            page,
            gr.update(choices=selection_choices, value=selected),
            gr.update(choices=album_choices(store), value=album) if update_filter
            else gr.update(),
            gr.update(choices=album_choices(store)),
            gr.update(choices=album_choices(store, include_unsorted=False,
                                            include_top_level=True)),
            gr.update(value=message),
        )

    # Every handler returns this same bundle, so wiring stays uniform.
    render_outputs = [gallery, page_info, tokens_state, selected_state, page_state,
                      selection_group, album_filter, target_album, new_album_parent, status]

    def refresh(album, page, selected, nested):
        return render(album, page, selected, nested=nested)

    def change_album(album, nested):
        # A different album means a fresh selection. The filter itself is left
        # untouched: it is the component that fired this event.
        return render(album, 1, [], nested=nested, update_filter=False)

    def toggle_nested(album, nested):
        return render(album, 1, [], nested=nested)

    def change_page(album, page, selected, nested, delta):
        return render(album, int(page or 1) + delta, selected, nested=nested)

    # -- selection --------------------------------------------------------
    def on_select(album, page, tokens, selected, nested, evt: gr.SelectData):
        album = album or UNSORTED
        visible = page_slice(list(tokens or []), int(page or 1), page_size)
        if evt.index is None or evt.index >= len(visible):
            return gr.update(), selected, gr.update(), gr.update(), gr.update()
        token = visible[evt.index]
        selected = list(selected or [])
        if token in selected:
            selected.remove(token)
        else:
            selected.append(token)
        slug, name = split_token(token)
        path = store.dir_for(slug) / name
        choices = [(caption_for(t, album, nested), t) for t in selected]
        return (
            gr.update(choices=choices, value=selected),
            selected,
            gr.update(value=str(path) if path.is_file() else None),
            gr.update(value=f"`{name}`" + (f" — {album_label(store, slug, full_path=True)}"
                                           if nested and slug != album else "")),
            gr.update(value=f"{len(selected)} selected"),
        )

    def on_checkbox_input(values, album, nested):
        # Fires only on real user interaction, so this cannot loop with the
        # programmatic updates made by on_select.
        values = list(values or [])
        choices = [(caption_for(t, album or UNSORTED, nested), t) for t in values]
        return gr.update(choices=choices, value=values), values, f"{len(values)} selected"

    def select_visible(album, page, tokens, selected, nested):
        visible = page_slice(list(tokens or []), int(page or 1), page_size)
        merged = list(dict.fromkeys(list(selected or []) + visible))
        return render(album, page, merged, f"Selected {len(visible)} on this page.",
                      nested=nested)

    def select_all(album, page, tokens, nested):
        tokens = list(tokens or [])
        return render(album, page, tokens, f"Selected all {len(tokens)} image(s).",
                      nested=nested)

    def clear_selection(album, page, nested):
        return render(album, page, [], "Selection cleared.", nested=nested)

    # -- album operations -------------------------------------------------
    def group_by_album(tokens: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for token in tokens:
            slug, name = split_token(token)
            grouped[slug].append(name)
        return grouped

    def assign(album, target, selected, page, nested):
        selected = list(selected or [])
        if not selected:
            return render(album, page, selected, "⚠️ Nothing selected.", nested=nested)
        if not target:
            return render(album, page, selected, "⚠️ Pick a destination album.",
                          nested=nested)
        grouped = group_by_album(selected)
        if set(grouped) == {target}:
            return render(album, page, selected, "⚠️ Those images are already there.",
                          nested=nested)
        moved = 0
        try:
            for source_slug, names in grouped.items():
                if source_slug == target:
                    continue
                moved += len(store.assign(names, source_slug, target))
        except StorageError as exc:
            return render(album, page, selected, f"❌ {exc}", nested=nested)
        message = (f"✅ Moved **{moved}** image(s) to "
                   f"**{album_label(store, target, full_path=True)}**.")
        return render(album, page, [], message, nested=nested)

    def create_and_assign(album, name, parent, selected, page, nested):
        selected = list(selected or [])
        if not (name or "").strip():
            return render(album, page, selected, "⚠️ Enter a name for the new album.",
                          nested=nested)
        try:
            created = store.create_album(name, parent=parent_value(parent))
        except StorageError as exc:
            return render(album, page, selected, f"❌ {exc}", nested=nested)
        where = store.path_name(created.slug)
        if not selected:
            return render(album, page, selected,
                          f"✅ Created **{where}** (nothing selected to move).",
                          nested=nested)
        moved = 0
        for source_slug, names in group_by_album(selected).items():
            moved += len(store.assign(names, source_slug, created.slug))
        return render(album, page, [],
                      f"✅ Created **{where}** and moved {moved} image(s).", nested=nested)

    def delete_selection(album, selected, page, confirm, nested):
        selected = list(selected or [])
        if not selected:
            return render(album, page, selected, "⚠️ Nothing selected.", nested=nested)
        if not confirm:
            return render(album, page, selected,
                          "⚠️ Tick the confirmation box to delete these images.",
                          nested=nested)
        removed = 0
        for source_slug, names in group_by_album(selected).items():
            removed += store.delete_images(names, source_slug)
        ctx.manifest.prune_missing()
        ctx.manifest.save_if_dirty()
        return render(album, page, [], f"🗑️ Deleted **{removed}** image(s).", nested=nested)

    # -- wiring -----------------------------------------------------------
    refresh_button.click(refresh,
                         inputs=[album_filter, page_state, selected_state, include_nested],
                         outputs=render_outputs)
    album_filter.change(change_album, inputs=[album_filter, include_nested],
                        outputs=render_outputs)
    include_nested.change(toggle_nested, inputs=[album_filter, include_nested],
                          outputs=render_outputs)
    prev_button.click(lambda a, p, s, n: change_page(a, p, s, n, -1),
                      inputs=[album_filter, page_state, selected_state, include_nested],
                      outputs=render_outputs)
    next_button.click(lambda a, p, s, n: change_page(a, p, s, n, 1),
                      inputs=[album_filter, page_state, selected_state, include_nested],
                      outputs=render_outputs)

    gallery.select(
        on_select,
        inputs=[album_filter, page_state, tokens_state, selected_state, include_nested],
        outputs=[selection_group, selected_state, preview, preview_caption, status],
    )
    selection_group.input(on_checkbox_input,
                          inputs=[selection_group, album_filter, include_nested],
                          outputs=[selection_group, selected_state, status])

    select_page_button.click(
        select_visible,
        inputs=[album_filter, page_state, tokens_state, selected_state, include_nested],
        outputs=render_outputs)
    select_all_button.click(select_all,
                            inputs=[album_filter, page_state, tokens_state, include_nested],
                            outputs=render_outputs)
    clear_button.click(clear_selection, inputs=[album_filter, page_state, include_nested],
                       outputs=render_outputs)

    assign_button.click(
        assign,
        inputs=[album_filter, target_album, selected_state, page_state, include_nested],
        outputs=render_outputs)
    create_button.click(
        create_and_assign,
        inputs=[album_filter, new_album_name, new_album_parent, selected_state, page_state,
                include_nested],
        outputs=render_outputs)
    delete_button.click(
        delete_selection,
        inputs=[album_filter, selected_state, page_state, delete_confirm, include_nested],
        outputs=render_outputs)

    return {
        "refresh": refresh,
        "outputs": render_outputs,
        "inputs": [album_filter, page_state, selected_state, include_nested],
        # Exposed so the handlers can be exercised directly in tests.
        "handlers": {
            "render": render,
            "change_album": change_album,
            "toggle_nested": toggle_nested,
            "select_visible": select_visible,
            "select_all": select_all,
            "clear_selection": clear_selection,
            "assign": assign,
            "create_and_assign": create_and_assign,
            "delete_selection": delete_selection,
            "view_tokens": view_tokens,
        },
    }
