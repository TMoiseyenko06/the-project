from __future__ import annotations

import pytest
from PIL import Image

from imagebatch.config import Config
from imagebatch.pipeline import EditPipeline, is_oom, prepare_image


# -- image preparation ----------------------------------------------------
def test_prepare_image_converts_mode() -> None:
    assert prepare_image(Image.new("L", (64, 64)), None).mode == "RGB"
    assert prepare_image(Image.new("RGBA", (64, 64)), None).mode == "RGB"


def test_prepare_image_downscales_to_max_side() -> None:
    result = prepare_image(Image.new("RGB", (2000, 1000)), max_side=512)
    assert max(result.size) <= 512


def test_prepare_image_does_not_upscale() -> None:
    assert prepare_image(Image.new("RGB", (128, 128)), max_side=1024).size == (128, 128)


def test_prepare_image_snaps_to_multiple() -> None:
    width, height = prepare_image(Image.new("RGB", (101, 203)), None, multiple=8).size
    assert width % 8 == 0 and height % 8 == 0


def test_prepare_image_preserves_aspect_roughly() -> None:
    result = prepare_image(Image.new("RGB", (1600, 800)), max_side=800)
    assert abs(result.size[0] / result.size[1] - 2.0) < 0.05


def test_prepare_tiny_image_stays_valid() -> None:
    assert prepare_image(Image.new("RGB", (3, 3)), None, multiple=8).size == (8, 8)


# -- oom classification ---------------------------------------------------
def test_is_oom_detects_message() -> None:
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_is_oom_ignores_other_errors() -> None:
    assert not is_oom(ValueError("bad prompt"))
    assert not is_oom(RuntimeError("shape mismatch"))


def test_is_oom_without_torch_installed() -> None:
    """Must not raise when torch is absent — it runs inside except blocks."""
    assert is_oom(RuntimeError("out of memory")) is True


# -- kwarg filtering ------------------------------------------------------
class NarrowPipe:
    def __call__(self, prompt=None, image=None, num_inference_steps=None):
        return type("Out", (), {"images": [image] if image else []})()


class WidePipe:
    def __call__(self, prompt=None, image=None, **kwargs):
        return type("Out", (), {"images": [image] if image else []})()


def make_loaded(config: Config, pipe) -> EditPipeline:
    pipeline = EditPipeline(config)
    pipeline.pipe = pipe
    pipeline._inspect_signature()
    return pipeline


def test_unsupported_kwargs_are_dropped() -> None:
    pipeline = make_loaded(Config(model_id="mock"), NarrowPipe())

    kwargs = pipeline.supported_kwargs(
        {"num_inference_steps": 20, "guidance_scale": 7.0, "image_guidance_scale": 1.5}
    )

    assert kwargs == {"num_inference_steps": 20}


def test_var_kwargs_pipeline_keeps_everything() -> None:
    pipeline = make_loaded(Config(model_id="mock"), WidePipe())
    kwargs = pipeline.supported_kwargs({"guidance_scale": 7.0, "strength": 0.4})
    assert kwargs == {"guidance_scale": 7.0, "strength": 0.4}


def test_image_kwarg_name_detection() -> None:
    class InitImagePipe:
        def __call__(self, prompt=None, init_image=None):
            return {"images": [init_image]}

    assert make_loaded(Config(model_id="mock"), InitImagePipe()).image_kwarg_name() == "init_image"
    assert make_loaded(Config(model_id="mock"), NarrowPipe()).image_kwarg_name() == "image"


def test_edit_through_narrow_pipeline() -> None:
    """A model that accepts only a subset of settings still runs."""
    config = Config(model_id="mock", guidance_scale=7.0, image_guidance_scale=1.5, max_side=64)
    pipeline = make_loaded(config, NarrowPipe())

    result = pipeline.edit([Image.new("RGB", (64, 64), "red")], "a prompt")

    assert len(result) == 1
    assert isinstance(result[0], Image.Image)


# -- output extraction ----------------------------------------------------
def test_extract_images_from_tuple() -> None:
    images = [Image.new("RGB", (8, 8))]
    assert EditPipeline._extract_images((images, None), expected=1) == images


def test_extract_images_from_dict() -> None:
    images = [Image.new("RGB", (8, 8))]
    assert EditPipeline._extract_images({"images": images}, expected=1) == images


def test_extract_single_image() -> None:
    image = Image.new("RGB", (8, 8))
    assert EditPipeline._extract_images(type("O", (), {"images": image})(), expected=1) == [image]


def test_extract_images_rejects_non_pil() -> None:
    with pytest.raises(RuntimeError, match="non-PIL"):
        EditPipeline._extract_images(type("O", (), {"images": [[0.1, 0.2]]})(), expected=1)


def test_extract_images_rejects_unknown_shape() -> None:
    with pytest.raises(RuntimeError, match="Could not find images"):
        EditPipeline._extract_images(object(), expected=1)


# -- mock pipeline --------------------------------------------------------
def test_mock_pipeline_roundtrip() -> None:
    pipeline = EditPipeline(Config(model_id="mock", max_side=64))
    pipeline.load()

    results = pipeline.edit([Image.new("RGB", (64, 64), "red")] * 3, "prompt")

    assert len(results) == 3
    assert all(isinstance(img, Image.Image) for img in results)


def test_describe_reports_state() -> None:
    pipeline = EditPipeline(Config(model_id="mock"))
    assert "not loaded" in pipeline.describe()
    pipeline.load()
    assert "mock" in pipeline.describe()
