"""Shared Hugging Face data must survive preparation of another model."""

from pathlib import Path

import pytest

from mlx_serve.config import settings
from mlx_serve.core.model_manager import model_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_original", [False, True])
async def test_conversion_preserves_shared_huggingface_cache(tmp_path, monkeypatch, keep_original):
    models = tmp_path / "models"
    models.mkdir()
    shared_cache = tmp_path / "hf" / "models--example--source" / "blobs"
    shared_cache.mkdir(parents=True)
    original = shared_cache / "shared-source"
    original.write_bytes(b"shared by other installations")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(model_manager, "_metadata_path", tmp_path / "metadata.json")
    monkeypatch.setattr(model_manager, "_metadata", {})

    async def convert(*, hf_repo, output_dir: Path, model_type, quantize):
        assert (hf_repo, model_type) == ("example/source", "embedding")
        output_dir.mkdir()
        (output_dir / "model.safetensors").write_bytes(b"converted")
        return True

    monkeypatch.setattr(model_manager, "_convert_model", convert)

    def do_not_scan_shared_cache(*args, **kwargs):
        raise AssertionError("Preparing a model must not delete shared cache revisions")

    monkeypatch.setattr("huggingface_hub.scan_cache_dir", do_not_scan_shared_cache)
    events = [
        event async for event in model_manager.pull_model(
            "example/source", "embedding", keep_original=keep_original
        )
    ]

    assert events[-1]["status"] == "success"
    assert original.read_bytes() == b"shared by other installations"
    assert model_manager._metadata["source"]["hf_repo"] == "example/source"
    assert all(event["status"] != "cleaning" for event in events)
