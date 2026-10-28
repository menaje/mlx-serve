"""Tests for importing an installed Ollama decision checkpoint."""

import errno
import hashlib
import json

import pytest

from mlx_serve.config import settings
from mlx_serve.core.model_manager import model_manager
from mlx_serve.core.ollama_import import import_ollama_decision


@pytest.fixture
def ollama_source(tmp_path, monkeypatch):
    source = tmp_path / "ollama"
    blobs = source / "blobs"
    blobs.mkdir(parents=True)
    files = {
        "config.json": b"{}",
        "decision_config.json": b'{"format_id":"d3-code-readout-v1"}',
        "tokenizer.json": b'{"tokenizer":true}',
        "tokenizer_config.json": b'{"settings":true}',
        "weight": b"readout",
        "language_model.embed_tokens.weight": b"text-weights",
    }
    layers = []
    for name, data in files.items():
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(data)
        layers.append(
            {
                "name": name,
                "digest": f"sha256:{digest}",
                "mediaType": "application/vnd.ollama.image.tensor"
                if name.endswith("weight")
                else "application/json",
            }
        )
    manifest = source / "manifests-v2" / "ollama.com" / "library" / "d3-lite" / "mlx"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"runner": "mlx", "format": "safetensors", "layers": layers}))
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(model_manager, "_metadata_path", tmp_path / "metadata.json")
    monkeypatch.setattr(model_manager, "_metadata", {})
    return source


def test_import_survives_deletion_of_ollama_blobs(ollama_source):
    destination = import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    assert model_manager._metadata["d3-lite"]["type"] == "decision"
    assert (destination / "readout.safetensors").read_bytes() == b"readout"
    assert (destination / "weights" / "0001.safetensors").read_bytes() == b"text-weights"
    for blob in (ollama_source / "blobs").iterdir():
        blob.unlink()
    assert (destination / "readout.safetensors").read_bytes() == b"readout"


def test_import_copies_if_hard_links_cross_filesystems(ollama_source, monkeypatch):
    def cross_device_link(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr("mlx_serve.core.ollama_import.os.link", cross_device_link)
    destination = import_ollama_decision("d3-lite:mlx", "copy", ollama_source)
    assert (destination / "readout.safetensors").read_bytes() == b"readout"


def test_import_does_not_replace_an_existing_model(ollama_source):
    destination = import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    with pytest.raises(FileExistsError):
        import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    assert (destination / "readout.safetensors").read_bytes() == b"readout"


def test_imported_model_can_be_deleted_by_ollama_alias(ollama_source):
    destination = import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    alias = "d3-lite:0.8b-mlx-bf16"
    assert model_manager.is_model_installed(alias)
    assert model_manager.delete_model(alias)
    assert not destination.exists()
    assert any((ollama_source / "blobs").iterdir())


def test_import_normalizes_same_model_alias_and_rejects_other_aliases(ollama_source):
    with pytest.raises(ValueError, match="reserved for"):
        import_ollama_decision("d3-lite:mlx", "d1-omni", ollama_source)
    destination = import_ollama_decision("d3-lite:mlx", "D3-LITE", ollama_source)
    assert destination.name == "d3-lite"
    assert model_manager.is_model_installed("D3-LITE")
    assert set(model_manager._metadata) == {"d3-lite"}


@pytest.mark.asyncio
async def test_pull_does_not_mix_downloaded_and_imported_weights(ollama_source):
    destination = import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    results = [status async for status in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert results[-1]["status"] == "error"
    assert "already exists" in results[-1]["message"]
    assert (destination / "readout.safetensors").read_bytes() == b"readout"
