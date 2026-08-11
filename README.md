# Batch Image Editor

Run a Hugging Face image-to-image editing model over hundreds of images with a
single prompt, then browse, review and organise the results into albums — all
from a web UI you can reach through vast.ai's port mapping.

- **Run** — point at a folder or upload a zip, type one prompt, pick where the
  results should land, watch progress.
- **Gallery** — thumbnail grid, filter by album, multi-select, file into albums.
- **Albums** — nest, rename, move, delete, download as zip.

Albums can contain albums, batches can file themselves into an album
automatically, and runs are resumable with per-image failure isolation and one
model load per process.

---

## 1. Install

```bash
git clone <your-repo-url> batch-image-editor
cd batch-image-editor

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**On vast.ai**, the image usually already ships a CUDA-matched `torch`. Installing
the pinned one from `requirements.txt` can replace it with a build that doesn't
match your driver. Prefer:

```bash
pip install gradio Pillow PyYAML diffusers transformers accelerate safetensors
# and only install torch yourself if it is genuinely missing:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Verify the install:

```bash
pytest -q          # 228 tests, no GPU required
```

## 2. Point it at your model

Two lines in `config.yaml` — this is the only change needed to swap models:

```yaml
model_id: timbrooks/instruct-pix2pix              # any HF repo id or local path
pipeline_class: StableDiffusionInstructPix2PixPipeline
```

`pipeline_class` is looked up on the `diffusers` package by name, so use whatever
class the model card names. Some common ones:

| Model family | `pipeline_class` |
| --- | --- |
| Generic img2img (SD / SDXL) | `AutoPipelineForImage2Image` |
| InstructPix2Pix | `StableDiffusionInstructPix2PixPipeline` |
| SDXL img2img | `StableDiffusionXLImg2ImgPipeline` |
| FLUX Kontext | `FluxKontextPipeline` |
| Qwen-Image-Edit | `QwenImageEditPipeline` |

Everything can also be set by environment variable with an `IMGBATCH_` prefix,
which beats the file:

```bash
export IMGBATCH_MODEL_ID=someorg/some-edit-model
export IMGBATCH_PIPELINE_CLASS=AutoPipelineForImage2Image
export IMGBATCH_DTYPE=bf16
```

### Generation settings are filtered, not forced

Models disagree about what they accept — `image_guidance_scale` exists on
InstructPix2Pix, `strength` on img2img, neither on some newer editors. The
pipeline's actual call signature is inspected at load time and unsupported
settings are **dropped with a log line** instead of raising. So you can leave
`config.yaml` as-is when swapping models, and only tune what matters. Set any
value to `none` to stop sending it at all.

Anything unusual your model wants goes through untouched:

```yaml
extra_pipeline_kwargs:
  true_cfg_scale: 4.0
```

### Fitting the GPU

```yaml
dtype: bf16                     # bf16 | fp16 | fp32 | auto
enable_attention_slicing: true  # lower VRAM, slightly slower
enable_vae_slicing: true
enable_xformers: true           # needs a matching xformers build
enable_model_cpu_offload: true  # last resort: fits big models, much slower
max_side: 1024                  # downscale huge inputs before inference
```

### Try it before the weights land

`model_id: mock` (the default) runs a placeholder transform with no GPU and no
downloads, so you can click through the whole UI — batching, albums, zips,
resume — before committing to a model. The Run tab shows a warning while mock
is active.

## 3. Launch

```bash
./run.sh                       # reads config.yaml, serves on 0.0.0.0:7860
./run.sh --port 8080
./run.sh --model someorg/some-model --preload
```

`run.sh` activates `.venv` if present (set `VENV=none` to use system python),
keeps the HF cache on the working disk, and `exec`s the app.

Run it inside **tmux** so it survives SSH disconnects:

```bash
tmux new -s editor
./run.sh
# detach: Ctrl-b then d
# reattach: tmux attach -t editor
```

By default the model loads lazily on the first batch. `--preload` loads it at
startup instead, so the first run isn't slowed by a cold load.

### Headless batch (no UI)

```bash
python app.py --cli /workspace/input_images "make it a pencil sketch"

# file the results straight into an album (created if it doesn't exist)
python app.py --cli /workspace/shoot_a "watercolour" --album "Edited"

# ...nested under another album
python app.py --cli /workspace/shoot_a "watercolour" \
    --album "Edited" --album-parent "Shoot A"
```

Same resume behaviour and same output layout; prints a summary and exits
non-zero if any image failed.

## 4. Reaching it on vast.ai

The app binds `0.0.0.0` so vast.ai's proxy can reach it. You need the port
**mapped when the instance is created** — it cannot be added afterwards.

1. When creating the instance, add the port under *Docker options* →
   `-p 7860:7860`, or set the env var `OPEN_BUTTON_PORT=7860`.
2. After it boots, the **Instance card → IP/Port** button lists the public
   mapping, e.g. `123.45.67.89:41234 -> 7860`.
3. Open `http://123.45.67.89:41234` in a browser.

If you skipped the mapping, tunnel over SSH instead — no instance rebuild needed:

```bash
ssh -p <ssh-port> -L 7860:localhost:7860 root@<instance-ip>
# then open http://localhost:7860 locally
```

As a last resort, `share: true` in `config.yaml` (or `--share`) creates a
temporary public `gradio.live` URL. It's convenient but publicly reachable by
anyone with the link, and there is no login on this app — prefer SSH tunnelling.

## 5. Using it

### Run tab
Choose a **folder path** on the instance or upload a **zip**. Type one prompt,
press *Run batch*. You get a live `X / N` count with ETA, and a summary listing
successes, skips and a table of failures. *Stop* halts after the current image;
whatever finished is kept and the next run picks up where it left off.

**Send results to** decides where this batch lands. Leave it on *Unsorted* for
the default behaviour, pick an existing album, or type a name under *…or create
a new album for this batch* to have one created (optionally nested under
another) and every image in the run filed into it as it is produced. A typed
name wins over the dropdown. Nothing is written if the name clashes with an
existing album — you get an error before any work starts.

### Gallery tab
Filter by album (or *Unsorted*). Tick **Include sub-albums** to see everything
nested underneath the selected album; images from sub-albums are captioned
`Sub-album / filename`. Click images to add them to the selection — the right
pane shows the clicked image full size, and the *Selection* list lets you untick
individual files. *Select page* / *Select all* / *Clear* for bulk work.

Then either move the selection to an existing album, or type a name, choose what
to nest it under, and *Create & move*. A selection spanning several sub-albums
moves correctly in one go. Deleting requires ticking the confirmation box.

Large batches are paginated (`gallery_page_size`, default 60) and the grid
renders cached thumbnails, so hundreds of images stay responsive.

### Albums tab
Cover thumbnails and an indented **album tree** with per-album counts, shown as
`direct (+nested)`. An album with no images of its own borrows a thumbnail from
a sub-album. Per album you can create a child, rename, **move** it under a
different parent (or back to the top level), download as zip, and delete.

The *Move under* list hides the album itself and its own descendants, so a cycle
can't be built from the UI; the store rejects one anyway if it's attempted.

Deleting an album:
- images go back to *Unsorted* unless you tick *delete the image files too*;
- sub-albums are **promoted** one level up unless you tick *delete its
  sub-albums too*, which removes the whole subtree.

*Unsorted* can be downloaded but not renamed, moved or deleted.

## 6. Output layout

```
outputs/
├── albums.json        # source of truth for album membership and nesting
├── manifest.json      # processed source images, for resuming
├── unsorted/          # where results land unless a batch targets an album
├── album_<slug>/      # one folder per album — flat, regardless of nesting
├── logs/              # run_<timestamp>.json, one per batch
└── .staging/          # extracted zips, thumbnail cache, prepared downloads
```

Moving an image between albums is a file move plus an `albums.json` update, done
under a lock and written atomically (temp file + rename), so an interrupted move
can't corrupt the index. If you rearrange files by hand, they're reconciled into
`albums.json` at the next startup.

### How nesting is stored

Each album records a `parent` slug in `albums.json`. The directories stay flat:
`Keepers / Portraits` lives in `outputs/album_portraits/`, not
`outputs/album_keepers/album_portraits/`. Re-parenting is then a single atomic
JSON write instead of a recursive directory move that could be interrupted
half-done, and renaming a parent never touches its children's paths.

The trade-off: **an `rsync` of `outputs/` gives you flat album folders** — the
hierarchy lives in `albums.json`, not in the directory names. Zip downloads do
rebuild it, so *Prepare zip* on `Keepers` produces:

```
img_0.png
Portraits/img_1.png
Portraits/Headshots/img_2.png
```

Untick *Include sub-albums* to zip just the album's own images.

An `albums.json` written before nesting existed loads fine — those albums become
top level. A hand-edited file with a missing parent or a cycle is repaired on
read rather than crashing.

Pull results off with the zip button, or directly:

```bash
rsync -avz -e "ssh -p <ssh-port>" root@<ip>:/path/to/outputs/ ./local-outputs/
```

## 7. Resuming

Every finished image is recorded in `manifest.json`, keyed by the source file's
**content hash** plus the prompt, model id and generation settings. Re-running
the same batch skips finished work; changing the prompt or the model reprocesses
it, because the key changes. Failures are recorded but always retried next run,
so fixing a corrupt file and re-running just picks it up.

The manifest is flushed as work completes, so a crash or an interrupted run
loses at most the image in flight. Filing results into albums doesn't break
resume — whether a batch wrote straight into an album or you moved the results
later, outputs are tracked to their current location. Deleting an output does
cause it to be regenerated.

Note that the destination album is *not* part of the resume key: re-running the
same prompt pointed at a different album skips the images rather than producing
a second copy elsewhere. Uncheck **Resume** if you want a fresh set.

Uncheck **Resume** in the Run tab (or `--no-resume` on the CLI) to force a full
reprocess.

## 8. Batch size

`batch_size: auto` times single vs batched inference on the first images of a
run and keeps whichever is faster per image; runs shorter than 8 images skip the
probe, since it costs two extra inferences. Pin an integer once you know what
fits:

```yaml
batch_size: 4
auto_batch_max: 4     # ceiling used when probing
```

If a batched call fails — OOM or a bad image — the whole group is retried one at
a time, so one bad image never costs you the other three.

## 9. Troubleshooting

**`diffusers has no pipeline class X`** — `pipeline_class` must be a class that
exists in your installed `diffusers`. Check the model card, and upgrade
`diffusers` for very new architectures.

**CUDA out of memory** — lower `max_side`, set `batch_size: 1`, enable
`enable_attention_slicing` / `enable_vae_slicing`, then `enable_model_cpu_offload`
as a last resort. Single-image OOM is caught, logged and skipped rather than
killing the run.

**`Could not find images on the pipeline output`** — the configured class isn't
an image-to-image pipeline, or it returned tensors. Try
`extra_pipeline_kwargs: {output_type: pil}`.

**Model downloads fill the disk** — `run.sh` sets `HF_HOME` to `./.hf_cache` in
the project directory. Point it at your big volume if that isn't the roomy one.

**Port already in use** — `./run.sh --port 7861`, or
`export IMGBATCH_PORT=7861`.

**Gallery images don't load** — the app tells Gradio to serve only `output_dir`.
If you moved outputs elsewhere mid-session, restart the app.

## 10. Layout

```
app.py                    entry point + argument parsing + headless CLI
config.yaml               all settings, with comments
run.sh                    tmux-friendly launcher
imagebatch/
  config.py               layered config: defaults < YAML < env
  pipeline.py             model loading, kwarg filtering, OOM handling
  batch.py                discovery, resume planning, progress, failure isolation
  storage.py              albums.json, file moves, zips
  manifest.py             resume records
  thumbnails.py           cached gallery thumbnails
  ui/                     Gradio layer, one module per tab
tests/                    228 tests, no GPU needed
```

Inference, storage and UI are separate layers: `storage.py` and `batch.py` have
no Gradio import, and the UI holds no inference logic.

## Notes

- No auth — anyone who can reach the port can use it. Keep it behind SSH.
- Tested against Gradio 4.44 and 5.50.
