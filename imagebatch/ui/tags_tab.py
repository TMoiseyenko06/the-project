"""Tags tab: browse images by tag, manage categories, name and merge faces."""

from __future__ import annotations

import logging

import gradio as gr

from ..storage import StorageError
from ..tags import UNTAGGED
from .context import AppContext

log = logging.getLogger(__name__)

NO_GROUPING = "— no grouping —"
# Gradio needs a fixed component tree, so group slots are pre-built and shown
# on demand. Searches with more groups than this are truncated with a note.
MAX_GROUPS = 12


def build_tags_tab(ctx: AppContext) -> dict:
    store = ctx.store

    gr.Markdown(
        "### Tags\n"
        "Search for a tag and group the results by another category — e.g. find "
        "every *Pose 1* image grouped by who's in it, or every *Alex* image "
        "grouped by pose."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                category_picker = gr.Dropdown(
                    choices=[], label="Category", interactive=True, scale=1)
                value_picker = gr.Dropdown(
                    choices=[], label="Value", interactive=True, scale=1)
                group_picker = gr.Dropdown(
                    choices=[NO_GROUPING], value=NO_GROUPING, label="Group by",
                    interactive=True, scale=1)
            with gr.Row():
                search_button = gr.Button("Search", variant="primary", scale=2)
                refresh_button = gr.Button("↻ Refresh", scale=1)
            results = gr.Markdown()
            galleries: list[tuple[gr.Markdown, gr.Gallery]] = []
            # Gradio needs a fixed component tree, so pre-build a pool of
            # group slots and show only as many as a given search fills.
            for _ in range(MAX_GROUPS):
                heading = gr.Markdown(visible=False)
                gallery = gr.Gallery(columns=5, height=260, object_fit="cover",
                                     allow_preview=False, show_label=False,
                                     visible=False)
                galleries.append((heading, gallery))

        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("**Categories**")
                new_category = gr.Textbox(label="New category",
                                          placeholder="e.g. outfit")
                create_category_button = gr.Button("Create category")
                delete_category_button = gr.Button("Delete selected category",
                                                   variant="stop")
            with gr.Group():
                gr.Markdown("**Faces**")
                gr.Markdown(
                    "_Faces detected during a run, grouped automatically. Give "
                    "them names here — the underlying id never changes, so "
                    "renaming can't break existing tags._")
                face_picker = gr.Dropdown(choices=[], label="Face", interactive=True)
                face_name = gr.Textbox(label="Name", placeholder="e.g. Alex")
                rename_face_button = gr.Button("Save name")
                with gr.Accordion("Merge two faces", open=False):
                    gr.Markdown("_Use when one person was split into two ids._")
                    merge_keep = gr.Dropdown(choices=[], label="Keep", interactive=True)
                    merge_absorb = gr.Dropdown(choices=[], label="Merge in (removed)",
                                               interactive=True)
                    merge_button = gr.Button("Merge", variant="stop")
            status = gr.Markdown()

    # -- helpers ------------------------------------------------------------
    def category_choices() -> list[tuple[str, str]]:
        return [(c["name"], c["key"]) for c in store.tags.list_categories()]

    def face_choices() -> list[tuple[str, str]]:
        return [(f"{f.label}  ({f.sample_count} photo(s))", f.face_id)
                for f in ctx.faces.list_faces()]

    def face_label(face_id: str) -> str:
        try:
            return ctx.faces.get(face_id).label
        except StorageError:
            return face_id

    def pretty_value(category: str, value: str) -> str:
        """Show a face's name rather than its raw id where one is set."""
        if category == ctx.config.face_category and value.startswith("face_"):
            return face_label(value)
        return value

    def refresh(selected_category=None, message=""):
        categories = category_choices()
        keys = [key for _, key in categories]
        if selected_category not in keys:
            selected_category = keys[0] if keys else None
        values = []
        if selected_category:
            try:
                values = [(pretty_value(selected_category, v), v)
                          for v in store.tags.list_values(selected_category)]
            except StorageError:
                values = []
        faces = face_choices()
        face_ids = [fid for _, fid in faces]
        return (
            gr.update(choices=categories, value=selected_category),
            gr.update(choices=values, value=values[0][1] if values else None),
            gr.update(choices=[NO_GROUPING] + categories, value=NO_GROUPING),
            gr.update(choices=faces, value=face_ids[0] if face_ids else None),
            gr.update(choices=faces, value=None),
            gr.update(choices=faces, value=None),
            gr.update(value=message),
        )

    controls = [category_picker, value_picker, group_picker, face_picker,
                merge_keep, merge_absorb, status]

    def on_category_change(category):
        values = []
        if category:
            try:
                values = [(pretty_value(category, v), v)
                          for v in store.tags.list_values(category)]
            except StorageError:
                values = []
        return gr.update(choices=values, value=values[0][1] if values else None)

    # -- search -------------------------------------------------------------
    def search(category, value, group_by):
        blanks = []
        for _ in range(MAX_GROUPS):
            blanks += [gr.update(visible=False), gr.update(visible=False)]

        if not category or not value:
            return [gr.update(value="⚠️ Pick a category and value first.")] + blanks

        group_key = None if group_by in (None, NO_GROUPING) else group_by
        try:
            buckets = store.tags.search(category, value, group_by=group_key)
        except StorageError as exc:
            return [gr.update(value=f"❌ {exc}")] + blanks

        if not buckets:
            return [gr.update(value="No images match that tag.")] + blanks

        total = sum(len(v) for v in buckets.values())
        label = pretty_value(category, value)
        header = f"**{total}** image(s) tagged `{category}={label}`"
        if group_key:
            header += f", grouped by **{store.tags.category_name(group_key)}**"
        if len(buckets) > MAX_GROUPS:
            header += (f"\n\n_Showing the first {MAX_GROUPS} of {len(buckets)} groups._")

        updates = []
        ordered = sorted(buckets.items(),
                         key=lambda kv: (kv[0] == UNTAGGED, kv[0].lower()))
        for i in range(MAX_GROUPS):
            if i < len(ordered):
                bucket, entries = ordered[i]
                items = []
                for album, filename in entries:
                    path = store.dir_for(album) / filename
                    if path.is_file():
                        items.append((ctx.thumbnails.get(path), filename))
                title = (pretty_value(group_key, bucket)
                         if group_key and bucket != UNTAGGED else bucket)
                heading = (f"#### {title} ({len(items)})" if group_key
                           else f"#### {len(items)} image(s)")
                updates += [gr.update(value=heading, visible=True),
                            gr.update(value=items, visible=True)]
            else:
                updates += [gr.update(visible=False), gr.update(visible=False)]
        return [gr.update(value=header)] + updates

    search_outputs = [results]
    for heading, gallery in galleries:
        search_outputs += [heading, gallery]

    # -- category management ------------------------------------------------
    def create_category(name):
        if not (name or "").strip():
            return refresh(None, "⚠️ Enter a category name.")
        try:
            key = store.tags.create_category(name)
        except StorageError as exc:
            return refresh(None, f"❌ {exc}")
        return refresh(key, f"✅ Created category **{name}**.")

    def delete_category(category):
        if not category:
            return refresh(None, "⚠️ Pick a category first.")
        name = store.tags.category_name(category)
        try:
            affected = store.tags.delete_category(category)
        except StorageError as exc:
            return refresh(category, f"❌ {exc}")
        return refresh(None, f"🗑️ Deleted **{name}** "
                             f"({affected} image(s) had it removed).")

    # -- faces ---------------------------------------------------------------
    def rename_face(face_id, name):
        if not face_id:
            return refresh(None, "⚠️ Pick a face first.")
        try:
            face = ctx.faces.rename(face_id, name)
        except StorageError as exc:
            return refresh(None, f"❌ {exc}")
        return refresh(None, f"✅ `{face.face_id}` is now **{face.label}**.")

    def merge_faces(keep_id, absorb_id):
        if not keep_id or not absorb_id:
            return refresh(None, "⚠️ Pick both faces to merge.")
        try:
            face = ctx.faces.merge(keep_id, absorb_id)
        except StorageError as exc:
            return refresh(None, f"❌ {exc}")
        # Re-point any images tagged with the absorbed id at the survivor, so
        # the merge is reflected in the tags and not just the face registry.
        category = ctx.config.face_category
        moved = 0
        for key, tags in store.tags.all_tagged_images().items():
            values = tags.get(category, [])
            if absorb_id not in values:
                continue
            album, filename = key.split("/", 1)
            updated = [keep_id if v == absorb_id else v for v in values]
            store.tags.set_tags(album, filename, category, updated)
            moved += 1
        return refresh(None, f"✅ Merged into **{face.label}** "
                             f"({moved} image(s) re-tagged).")

    # -- wiring ---------------------------------------------------------------
    refresh_button.click(lambda c: refresh(c), inputs=category_picker, outputs=controls)
    category_picker.change(on_category_change, inputs=category_picker,
                           outputs=value_picker)
    search_button.click(search, inputs=[category_picker, value_picker, group_picker],
                        outputs=search_outputs)
    create_category_button.click(create_category, inputs=new_category, outputs=controls)
    delete_category_button.click(delete_category, inputs=category_picker,
                                 outputs=controls)
    rename_face_button.click(rename_face, inputs=[face_picker, face_name],
                             outputs=controls)
    merge_button.click(merge_faces, inputs=[merge_keep, merge_absorb], outputs=controls)

    return {
        "refresh": lambda c=None: refresh(c),
        "outputs": controls,
        "inputs": [category_picker],
        "handlers": {
            "refresh": refresh, "search": search, "create_category": create_category,
            "delete_category": delete_category, "rename_face": rename_face,
            "merge_faces": merge_faces, "on_category_change": on_category_change,
            "pretty_value": pretty_value,
        },
    }
