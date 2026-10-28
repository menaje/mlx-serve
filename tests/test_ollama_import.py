"""Tests for importing an installed Ollama decision checkpoint."""

import asyncio
import errno
import hashlib
import json
import threading
from pathlib import Path

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


@pytest.mark.asyncio
async def test_failed_d3_download_can_be_retried(ollama_source, monkeypatch):
    attempts = 0

    def download(*, repo_id, local_dir):
        nonlocal attempts
        attempts += 1
        (Path(local_dir) / "weights").write_text("partial" if attempts == 1 else "complete")
        if attempts == 1:
            raise OSError("download interrupted")

    monkeypatch.setattr("mlx_serve.core.decision_prepare.snapshot_download", download)
    first = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert first[-1]["status"] == "error"
    assert not (settings.models_dir / "d3-lite").exists()
    assert list(settings.models_dir.iterdir()) == []
    assert "d3-lite" not in model_manager._metadata

    second = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert second[-1]["status"] == "success"
    assert (settings.models_dir / "d3-lite" / "weights").read_text() == "complete"
    assert model_manager._metadata["d3-lite"]["type"] == "decision"


@pytest.mark.asyncio
async def test_d3_registration_failure_removes_only_the_new_download(ollama_source, monkeypatch):
    settings.models_dir.mkdir()
    existing = settings.models_dir / "other"
    existing.mkdir()
    (existing / "weights").write_text("preserved")

    def download(*, repo_id, local_dir):
        (Path(local_dir) / "weights").write_text("new")

    def fail_save():
        raise OSError("cannot save registry")

    monkeypatch.setattr("mlx_serve.core.decision_prepare.snapshot_download", download)
    monkeypatch.setattr(model_manager, "_save_metadata", fail_save)
    events = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert events[-1]["status"] == "error"
    assert not (settings.models_dir / "d3-lite").exists()
    assert (existing / "weights").read_text() == "preserved"


@pytest.mark.asyncio
async def test_cancelled_d3_download_waits_for_thread_before_cleaning_stage(
    ollama_source, monkeypatch
):
    started = threading.Event()
    release = threading.Event()

    def download(*, repo_id, local_dir):
        started.set()
        release.wait(timeout=5)
        (Path(local_dir) / "weights").write_text("complete")

    monkeypatch.setattr("mlx_serve.core.decision_prepare.snapshot_download", download)
    events = model_manager.pull_model("vllm-sr/d3-lite", "decision")
    assert (await anext(events))["status"] == "downloading"
    pending = asyncio.create_task(anext(events))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        pending.cancel()
        await asyncio.sleep(0.05)
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert list(settings.models_dir.iterdir()) == []
    finally:
        release.set()
