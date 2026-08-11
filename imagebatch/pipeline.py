"""Model loading and image-to-image inference.

The pipeline class is resolved by name from ``diffusers`` at runtime, so
switching models is a config change (``model_id`` + ``pipeline_class``) rather
than a code change. Call kwargs are filtered against the pipeline's actual
signature, so settings a given model doesn't accept are dropped instead of
raising.
"""

from __future__ import annotations

import gc
import inspect
import logging
import threading
import time
from typing import Any, Sequence

from PIL import Image

from .config import DTYPE_ALIASES, Config

log = logging.getLogger(__name__)


class OutOfMemoryError(RuntimeError):
    """Raised when inference runs out of GPU memory (caller may retry smaller)."""


def _torch():
    import torch  # imported lazily so the storage/UI layers work without CUDA

    return torch


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    log.warning("No GPU detected; falling back to CPU (this will be slow)")
    return "cpu"


def resolve_dtype(name: str, device: str) -> Any:
    torch = _torch()
    alias = DTYPE_ALIASES[name]
    if alias == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if device == "cpu" and alias in ("float16", "bfloat16"):
        # Half precision on CPU is either unsupported or pathologically slow.
        log.warning("dtype %s is not usable on CPU; using float32", name)
        return torch.float32
    return getattr(torch, alias)


def is_oom(exc: BaseException) -> bool:
    """Classify an exception as a GPU out-of-memory error.

    Must never raise: it runs inside an ``except`` block, and torch may not even
    be installed (the mock pipeline runs fine without it).
    """
    try:
        torch = _torch()
    except ImportError:
        pass
    else:
        cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if cuda_oom is not None and isinstance(exc, cuda_oom):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def prepare_image(image: Image.Image, max_side: int | None, multiple: int = 8) -> Image.Image:
    """Convert to RGB and fit within ``max_side``, snapped to a valid multiple."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    if max_side and max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        width, height = int(width * scale), int(height * scale)
    if multiple > 1:
        width = max(multiple, (width // multiple) * multiple)
        height = max(multiple, (height // multiple) * multiple)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.LANCZOS)
    return image


class MockPipeline:
    """Deterministic stand-in used when ``model_id: mock``.

    Lets the batch runner, album storage and UI be exercised end to end (and
    tested) on a machine with no GPU and no weights downloaded.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def __call__(self, prompt: Any = None, image: Any = None, **kwargs: Any) -> Any:
        from PIL import ImageOps

        images = image if isinstance(image, list) else [image]
        if self.delay:
            time.sleep(self.delay * len(images))
        edited = [ImageOps.autocontrast(ImageOps.mirror(img.convert("RGB"))) for img in images]
        return type("MockOutput", (), {"images": edited})()


class EditPipeline:
    """Loads the model once and keeps it resident for the process lifetime."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pipe: Any = None
        self.device: str = "cpu"
        self.dtype: Any = None
        self._lock = threading.RLock()
        self._call_params: set[str] | None = None
        self._accepts_var_kwargs = False

    # -- loading ----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        """Load weights onto the device. Safe to call repeatedly (no-op if loaded)."""
        with self._lock:
            if self.pipe is not None:
                return
            cfg = self.config
            if cfg.is_mock:
                log.warning("Using MockPipeline (model_id='mock') — no real editing happens")
                self.pipe = MockPipeline()
                self.device = "cpu"
                self._inspect_signature()
                return

            import diffusers

            if not hasattr(diffusers, cfg.pipeline_class):
                raise AttributeError(
                    f"diffusers has no pipeline class {cfg.pipeline_class!r}. "
                    "Set `pipeline_class` to the class documented on the model card "
                    "(e.g. AutoPipelineForImage2Image, StableDiffusionInstructPix2PixPipeline)."
                )
            pipeline_cls = getattr(diffusers, cfg.pipeline_class)

            self.device = resolve_device(cfg.device)
            self.dtype = resolve_dtype(cfg.dtype, self.device)

            kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
            if cfg.variant:
                kwargs["variant"] = cfg.variant
            if cfg.revision:
                kwargs["revision"] = cfg.revision
            if not cfg.safety_checker:
                # Only meaningful for pipelines that have one; harmless otherwise.
                kwargs["safety_checker"] = None

            log.info("Loading %s from %r (dtype=%s, device=%s)",
                     cfg.pipeline_class, cfg.model_id, cfg.dtype, self.device)
            started = time.time()
            try:
                pipe = pipeline_cls.from_pretrained(cfg.model_id, **kwargs)
            except TypeError as exc:
                # Some pipelines reject safety_checker/variant; retry without them.
                log.warning("from_pretrained rejected optional kwargs (%s); retrying minimally", exc)
                pipe = pipeline_cls.from_pretrained(cfg.model_id, torch_dtype=self.dtype)

            if cfg.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                # Offload manages placement itself; calling .to(device) would undo it.
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)

            self._apply_loras(pipe)
            self._apply_memory_options(pipe)
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)

            self.pipe = pipe
            self._inspect_signature()
            log.info("Model ready in %.1fs", time.time() - started)

    def _apply_loras(self, pipe: Any) -> None:
        """Load and activate the configured LoRA adapters, if any.

        Applied once at load time (not per call): diffusers bakes the adapter
        into the pipeline's PEFT layers via ``load_lora_weights`` /
        ``set_adapters``, so every subsequent inference in the batch already
        uses it.
        """
        loras = self.config.loras
        if not loras:
            return
        if not hasattr(pipe, "load_lora_weights"):
            raise AttributeError(
                f"{self.config.pipeline_class} has no load_lora_weights method, so "
                "it does not support the `loras` you configured. Remove them, or "
                "switch to a pipeline that supports LoRA."
            )
        names: list[str] = []
        scales: list[float] = []
        for i, lora in enumerate(loras):
            adapter_name = str(lora.get("adapter_name") or f"lora_{i}")
            kwargs: dict[str, Any] = {"adapter_name": adapter_name}
            if lora.get("weight_name"):
                kwargs["weight_name"] = lora["weight_name"]
            log.info("Loading LoRA %r as adapter %r", lora["repo_id"], adapter_name)
            pipe.load_lora_weights(lora["repo_id"], **kwargs)
            names.append(adapter_name)
            scales.append(float(lora.get("scale", 1.0)))
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(names, adapter_weights=scales)
        else:
            log.warning("Pipeline has no set_adapters; LoRA `scale` settings were ignored "
                        "(adapters loaded at their default weight)")

    def _apply_memory_options(self, pipe: Any) -> None:
        cfg = self.config
        toggles = [
            (cfg.enable_attention_slicing, "enable_attention_slicing"),
            (cfg.enable_vae_slicing, "enable_vae_slicing"),
            (cfg.enable_xformers, "enable_xformers_memory_efficient_attention"),
        ]
        for enabled, method_name in toggles:
            if not enabled:
                continue
            method = getattr(pipe, method_name, None)
            if method is None:
                log.warning("Pipeline does not support %s; skipping", method_name)
                continue
            try:
                method()
                log.info("Enabled %s", method_name)
            except Exception as exc:  # noqa: BLE001 - optional optimisation
                log.warning("%s failed (%s); continuing without it", method_name, exc)

    def _inspect_signature(self) -> None:
        """Record which kwargs the loaded pipeline's ``__call__`` accepts."""
        try:
            signature = inspect.signature(self.pipe.__call__)
        except (TypeError, ValueError):
            self._call_params = None
            self._accepts_var_kwargs = True
            return
        params = signature.parameters
        self._accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        self._call_params = {
            name for name, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        log.debug("Pipeline accepts kwargs: %s", sorted(self._call_params or []))

    def supported_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs the loaded pipeline does not accept."""
        if self._call_params is None or self._accepts_var_kwargs:
            # A **kwargs pipeline forwards anything; filtering would be wrong.
            return dict(kwargs)
        supported, dropped = {}, []
        for key, value in kwargs.items():
            if key in self._call_params:
                supported[key] = value
            else:
                dropped.append(key)
        if dropped:
            # Not an error: models differ in which knobs they expose.
            log.info("Pipeline ignores unsupported kwargs: %s", ", ".join(sorted(dropped)))
        return supported

    def image_kwarg_name(self) -> str:
        """Name of the input-image parameter (``image`` for nearly all pipelines)."""
        if self._call_params:
            for candidate in ("image", "images", "init_image"):
                if candidate in self._call_params:
                    return candidate
        return "image"

    # -- inference --------------------------------------------------------
    def edit(self, images: Sequence[Image.Image], prompt: str,
             seed: int | None = None) -> list[Image.Image]:
        """Run the pipeline over one or more images and return the results."""
        if not self.loaded:
            self.load()
        if not images:
            return []

        cfg = self.config
        prepared = [prepare_image(img, cfg.max_side, cfg.resize_multiple) for img in images]
        kwargs = self.supported_kwargs(cfg.generation_kwargs())

        prompts: Any = [prompt] * len(prepared) if len(prepared) > 1 else prompt
        image_arg: Any = prepared if len(prepared) > 1 else prepared[0]
        if "negative_prompt" in kwargs and len(prepared) > 1:
            kwargs["negative_prompt"] = [kwargs["negative_prompt"]] * len(prepared)

        seed = cfg.seed if seed is None else seed
        wants_generator = (
            self._call_params is None
            or "generator" in self._call_params
            or self._accepts_var_kwargs
        )
        if seed is not None and wants_generator and not cfg.is_mock:
            torch = _torch()
            device = "cpu" if self.device == "mps" else self.device
            if len(prepared) > 1:
                kwargs["generator"] = [
                    torch.Generator(device=device).manual_seed(seed + i)
                    for i in range(len(prepared))
                ]
            else:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        call_kwargs = {"prompt": prompts, self.image_kwarg_name(): image_arg, **kwargs}

        with self._lock:  # a single GPU pipeline is not safe to call concurrently
            try:
                result = self._invoke(call_kwargs)
            except Exception as exc:  # noqa: BLE001 - classified and re-raised below
                if is_oom(exc):
                    self.free_memory()
                    raise OutOfMemoryError(str(exc)) from exc
                raise

        return self._extract_images(result, expected=len(prepared))

    def _invoke(self, call_kwargs: dict[str, Any]) -> Any:
        if self.config.is_mock:
            return self.pipe(**call_kwargs)
        torch = _torch()
        with torch.inference_mode():
            return self.pipe(**call_kwargs)

    @staticmethod
    def _extract_images(result: Any, expected: int) -> list[Image.Image]:
        """Pull PIL images out of whatever shape the pipeline returned."""
        images = getattr(result, "images", None)
        if images is None and isinstance(result, (tuple, list)) and result:
            images = result[0]
        if images is None and isinstance(result, dict):
            images = result.get("images")
        if images is None:
            raise RuntimeError(
                "Could not find images on the pipeline output "
                f"(type {type(result).__name__}). This pipeline may not be an "
                "image-to-image pipeline."
            )
        if isinstance(images, Image.Image):
            images = [images]
        images = list(images)
        if not all(isinstance(img, Image.Image) for img in images):
            raise RuntimeError(
                "Pipeline returned non-PIL output; set output_type='pil' via "
                "extra_pipeline_kwargs for this model."
            )
        if len(images) != expected:
            log.warning("Pipeline returned %d images for %d inputs", len(images), expected)
        return images

    # -- housekeeping -----------------------------------------------------
    def free_memory(self) -> None:
        gc.collect()
        if self.device.startswith("cuda"):
            try:
                _torch().cuda.empty_cache()
            except Exception:  # noqa: BLE001 - cleanup must never mask the real error
                pass

    def describe(self) -> str:
        cfg = self.config
        if not self.loaded:
            return f"{cfg.model_id} (not loaded)"
        where = self.device if not cfg.enable_model_cpu_offload else f"{self.device} + cpu offload"
        suffix = f" · {len(cfg.loras)} LoRA(s)" if cfg.loras else ""
        return f"{cfg.model_id} · {cfg.pipeline_class} · {cfg.dtype} · {where}{suffix}"

    def unload(self) -> None:
        with self._lock:
            self.pipe = None
            self._call_params = None
            self.free_memory()
