from __future__ import annotations

from pathlib import Path

import pytest

from imagebatch.config import Config, load_config


def test_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.port == 7860
    assert config.host == "0.0.0.0"
    assert config.batch_size == "auto"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model_id: someorg/some-edit-model\n"
        "pipeline_class: StableDiffusionInstructPix2PixPipeline\n"
        "dtype: fp16\n"
        "port: 9000\n"
        "batch_size: 4\n"
    )

    config = load_config(path)

    assert config.model_id == "someorg/some-edit-model"
    assert config.pipeline_class == "StableDiffusionInstructPix2PixPipeline"
    assert config.dtype == "fp16"
    assert config.port == 9000
    assert config.batch_size == 4


def test_env_overrides_yaml(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model_id: from-yaml\nport: 9000\n")
    monkeypatch.setenv("IMGBATCH_MODEL_ID", "from-env")
    monkeypatch.setenv("IMGBATCH_PORT", "7777")

    config = load_config(path)

    assert config.model_id == "from-env"
    assert config.port == 7777


def test_env_type_coercion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMGBATCH_ENABLE_XFORMERS", "true")
    monkeypatch.setenv("IMGBATCH_SAFETY_CHECKER", "0")
    monkeypatch.setenv("IMGBATCH_GUIDANCE_SCALE", "3.5")
    monkeypatch.setenv("IMGBATCH_BATCH_SIZE", "auto")
    monkeypatch.setenv("IMGBATCH_SEED", "42")

    config = load_config()

    assert config.enable_xformers is True
    assert config.safety_checker is False
    assert config.guidance_scale == 3.5
    assert config.batch_size == "auto"
    assert config.seed == 42


def test_env_batch_size_int(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMGBATCH_BATCH_SIZE", "8")
    assert load_config().batch_size == 8


def test_none_disables_optional_kwarg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMGBATCH_IMAGE_GUIDANCE_SCALE", "none")
    config = load_config()
    assert config.image_guidance_scale is None
    assert "image_guidance_scale" not in config.generation_kwargs()


def test_unknown_keys_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model_id: x\nnot_a_real_key: 1\n")
    assert load_config(path).model_id == "x"


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_invalid_dtype_rejected() -> None:
    with pytest.raises(ValueError, match="dtype"):
        Config(dtype="float8")


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size="huge")
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size=0)


def test_generation_kwargs_drops_none() -> None:
    config = Config(strength=None, negative_prompt=None, guidance_scale=7.0)
    kwargs = config.generation_kwargs()
    assert "strength" not in kwargs
    assert "negative_prompt" not in kwargs
    assert kwargs["guidance_scale"] == 7.0


def test_extra_pipeline_kwargs_passthrough() -> None:
    config = Config(extra_pipeline_kwargs={"true_cfg_scale": 4.0})
    assert config.generation_kwargs()["true_cfg_scale"] == 4.0


def test_signature_changes_with_settings() -> None:
    base = Config(model_id="a", guidance_scale=7.0)
    assert base.signature() != Config(model_id="b", guidance_scale=7.0).signature()
    assert base.signature() != Config(model_id="a", guidance_scale=3.0).signature()
    assert base.signature() == Config(model_id="a", guidance_scale=7.0).signature()


def test_signature_ignores_non_visual_settings() -> None:
    """Changing the port must not invalidate finished work."""
    assert Config(port=1).signature() == Config(port=2).signature()
