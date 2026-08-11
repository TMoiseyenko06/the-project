"""Configuration loading.

Precedence (later wins): built-in defaults < YAML config file < environment
variables. Every field can be set with an ``IMGBATCH_``-prefixed env var, e.g.
``IMGBATCH_MODEL_ID=stabilityai/sdxl-refiner-1.0``.

Swapping models is a config change, not a code change: ``model_id`` picks the
weights and ``pipeline_class`` picks the ``diffusers`` class used to load them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ENV_PREFIX = "IMGBATCH_"
DEFAULT_CONFIG_PATH = "config.yaml"

# Values accepted for `dtype`, mapped to torch attribute names. Resolved lazily
# in pipeline.py so that importing this module never requires torch.
DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "auto": "auto",
}


@dataclass
class Config:
    # --- model -----------------------------------------------------------
    # Any diffusers-style repo id or local path. "mock" runs a no-GPU stub
    # pipeline, useful for exercising the UI before real weights are dropped in.
    model_id: str = "mock"
    # Name of the class to load the weights with. Resolved via
    # getattr(diffusers, pipeline_class), so any diffusers pipeline works.
    pipeline_class: str = "AutoPipelineForImage2Image"
    variant: str | None = None  # e.g. "fp16" for repos that ship fp16 weights
    revision: str | None = None
    dtype: str = "bf16"  # bf16 | fp16 | fp32 | auto
    device: str = "auto"  # auto | cuda | cuda:1 | cpu | mps
    # Memory-saving switches; all are best-effort and skipped when unsupported.
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_xformers: bool = False
    # Offloads whole pipeline components (text encoder / transformer / VAE)
    # between GPU and CPU as each takes its turn. A single large transformer is
    # still one component, so this doesn't shrink its own footprint — it only
    # avoids the other components also being resident at the same time.
    enable_model_cpu_offload: bool = False
    # Offloads at the individual-layer level instead, so even one huge
    # transformer only needs one layer's worth of weights on GPU at a time.
    # Much slower than enable_model_cpu_offload, but the real fix when a single
    # component doesn't fit in VRAM on its own. Mutually exclusive with
    # enable_model_cpu_offload.
    enable_sequential_cpu_offload: bool = False
    safety_checker: bool = True  # False disables it when the pipeline has one

    # --- generation ------------------------------------------------------
    # Extra kwargs forwarded to the pipeline call. Unsupported keys are dropped
    # automatically after inspecting the pipeline signature, so the same config
    # works across models with different call APIs.
    num_inference_steps: int | None = 30
    guidance_scale: float | None = 7.5
    image_guidance_scale: float | None = 1.5  # instruct-pix2pix style models
    strength: float | None = None  # img2img style models
    negative_prompt: str | None = None
    seed: int | None = None  # None => nondeterministic
    extra_pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    # LoRA adapters applied on top of the base model after it loads. Each entry
    # needs `repo_id`; `weight_name` is only required if that repo hosts more
    # than one weights file, `scale` defaults to 1.0, `adapter_name` is
    # auto-generated if omitted. Ignored while model_id is "mock".
    loras: list[dict[str, Any]] = field(default_factory=list)

    # --- batching / images ----------------------------------------------
    # "auto" benchmarks single vs batched inference on the first images of a run
    # and keeps whichever is faster. An integer pins the batch size.
    batch_size: str | int = "auto"
    auto_batch_max: int = 4
    max_side: int | None = 1024  # longest edge; None keeps original size
    resize_multiple: int = 8  # most UNets need dimensions divisible by 8
    output_format: str = "png"  # png | jpg | webp
    output_quality: int = 95  # jpg/webp only

    # --- storage ---------------------------------------------------------
    output_dir: str = "outputs"

    # --- server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    gallery_page_size: int = 60
    thumbnail_size: int = 384

    # --- misc ------------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_ALIASES:
            raise ValueError(
                f"dtype must be one of {sorted(DTYPE_ALIASES)}, got {self.dtype!r}"
            )
        if self.output_format not in ("png", "jpg", "jpeg", "webp"):
            raise ValueError(f"unsupported output_format: {self.output_format!r}")
        if isinstance(self.batch_size, str) and self.batch_size != "auto":
            raise ValueError("batch_size must be an int or 'auto'")
        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.auto_batch_max < 1:
            raise ValueError("auto_batch_max must be >= 1")
        for entry in self.loras:
            if not isinstance(entry, dict) or not str(entry.get("repo_id") or "").strip():
                raise ValueError(
                    f"each entry in `loras` needs a repo_id, got {entry!r}"
                )
        if self.enable_model_cpu_offload and self.enable_sequential_cpu_offload:
            raise ValueError(
                "enable_model_cpu_offload and enable_sequential_cpu_offload are "
                "mutually exclusive — sequential offload is the stronger of the "
                "two, so turn off enable_model_cpu_offload if you want it"
            )

    # -- derived helpers --------------------------------------------------
    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def is_mock(self) -> bool:
        return self.model_id.strip().lower() == "mock"

    def generation_kwargs(self) -> dict[str, Any]:
        """Candidate kwargs for a pipeline call, before signature filtering."""
        candidates = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_guidance_scale": self.image_guidance_scale,
            "strength": self.strength,
            "negative_prompt": self.negative_prompt,
        }
        kwargs = {k: v for k, v in candidates.items() if v is not None}
        kwargs.update(self.extra_pipeline_kwargs)
        return kwargs

    def signature(self) -> dict[str, Any]:
        """Identity of the settings that affect output pixels.

        Stored per image in the manifest so that changing the model or the
        generation settings invalidates previous results instead of silently
        being skipped as "already done".
        """
        return {
            "model_id": self.model_id,
            "pipeline_class": self.pipeline_class,
            "dtype": self.dtype,
            "max_side": self.max_side,
            "loras": self.loras,
            **self.generation_kwargs(),
        }


def _coerce(raw: str, target_type: Any) -> Any:
    """Convert an env var string into the type declared on the dataclass."""
    text = raw.strip()
    if text.lower() in ("none", "null"):
        # Explicitly unset an optional setting, e.g. IMGBATCH_STRENGTH=none to
        # stop passing `strength` to a pipeline that doesn't want it.
        return None
    if target_type is bool or target_type == "bool":
        return text.lower() in ("1", "true", "yes", "on")
    if target_type in (int, "int"):
        return int(text)
    if target_type in (float, "float"):
        return float(text)
    if target_type in (dict, "dict") or text.startswith("{"):
        return yaml.safe_load(text)
    return text


def _field_type(name: str) -> Any:
    """Best-effort scalar type for a field, looking through optionals/unions."""
    for f in fields(Config):
        if f.name != name:
            continue
        annotation = str(f.type)
        if "dict" in annotation:
            return dict
        if "bool" in annotation:
            return bool
        # batch_size is `str | int`: keep ints as ints, leave "auto" a string.
        if name == "batch_size":
            return "batch_size"
        if "int" in annotation:
            return int
        if "float" in annotation:
            return float
        return str
    raise KeyError(name)


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any], source: str) -> None:
    known = {f.name for f in fields(Config)}
    for key, value in overrides.items():
        if key not in known:
            log.warning("Ignoring unknown config key %r from %s", key, source)
            continue
        data[key] = value


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a :class:`Config` from defaults, an optional YAML file and env vars."""
    data: dict[str, Any] = {}

    config_path = Path(path or os.environ.get(f"{ENV_PREFIX}CONFIG", DEFAULT_CONFIG_PATH))
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
        _apply_overrides(data, loaded, str(config_path))
        log.info("Loaded config from %s", config_path)
    elif path is not None:
        raise FileNotFoundError(f"config file not found: {config_path}")

    for f in fields(Config):
        env_key = f"{ENV_PREFIX}{f.name.upper()}"
        if env_key in os.environ:
            ftype = _field_type(f.name)
            raw = os.environ[env_key]
            if ftype == "batch_size":
                data[f.name] = raw if raw.strip() == "auto" else int(raw)
            else:
                data[f.name] = _coerce(raw, ftype)

    if overrides:
        _apply_overrides(data, overrides, "runtime overrides")

    # "none"/"null" in YAML or env means "leave this kwarg out entirely".
    for key, value in list(data.items()):
        if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
            if key in ("negative_prompt", "variant", "revision", "seed", "strength",
                       "max_side", "image_guidance_scale", "guidance_scale",
                       "num_inference_steps"):
                data[key] = None

    return Config(**data)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
