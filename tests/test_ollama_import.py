"""Tests for importing an installed Ollama decision checkpoint."""

import asyncio
import errno
import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_serve.config import settings
from mlx_serve.core.decision_prepare import read_verified_manifest
from mlx_serve.core.decision_prepare_worker import D3_REPO, MANIFEST, file_hash, prepare_d3
from mlx_serve.core.model_manager import TTLLRUCache, model_manager
from mlx_serve.core.ollama_import import import_ollama_decision

D3_REVISION = "0123456789abcdef0123456789abcdef01234567"


def create_d3_artifact(path: Path) -> None:
    path.mkdir(parents=True)
    for name, data in {
        "config.json": b"{}", "decision_config.json": b"{}", "tokenizer.json": b"{}",
        "readout.safetensors": b"readout", "model.safetensors": b"complete",
    }.items():
        (path / name).write_bytes(data)
    manifest = {
        "schema_version": 1, "source_repo": D3_REPO, "revision": D3_REVISION,
        "architecture": "qwen3_5", "precision": "bfloat16",
        "files_sha256": {p.name: file_hash(p) for p in path.iterdir()},
        "validation": {"mlx_predict": True, "published_file_hashes": True},
    }
    (path / MANIFEST).write_text(json.dumps(manifest))


def mock_model_info(monkeypatch):
    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.HfApi.model_info",
        lambda *args, **kwargs: SimpleNamespace(sha=D3_REVISION, siblings=[]),
    )


def fake_worker(returncode=0, output=b""):
    reader = asyncio.StreamReader()
    reader.feed_data(output)
    reader.feed_eof()
    return SimpleNamespace(
        stdout=reader, pid=os.getpid(), returncode=returncode,
        wait=lambda: asyncio.sleep(0, result=returncode),
    )


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
    mock_model_info(monkeypatch)
    attempts = 0

    async def worker(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return fake_worker(
                1, b'MLX_PREP_STATUS={"status":"error","message":"download interrupted"}\n'
            )
        create_d3_artifact(Path(args[-1]))
        return fake_worker()

    monkeypatch.setattr("mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", worker)
    first = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert first[-1]["status"] == "error"
    assert not (settings.models_dir / "d3-lite").exists()
    assert not list(settings.models_dir.glob(".decision-prepare-*"))
    assert "d3-lite" not in model_manager._metadata

    second = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert second[-1]["status"] == "success"
    assert (settings.models_dir / "d3-lite" / "model.safetensors").read_bytes() == b"complete"
    assert model_manager._metadata["d3-lite"]["type"] == "decision"
    assert model_manager._metadata["d3-lite"]["revision"] == D3_REVISION


@pytest.mark.asyncio
async def test_d3_registration_failure_removes_only_the_new_download(ollama_source, monkeypatch):
    mock_model_info(monkeypatch)
    settings.models_dir.mkdir()
    existing = settings.models_dir / "other"
    existing.mkdir()
    (existing / "weights").write_text("preserved")

    async def worker(*args, **kwargs):
        create_d3_artifact(Path(args[-1]))
        return fake_worker()

    def fail_save():
        raise OSError("cannot save registry")

    monkeypatch.setattr("mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", worker)
    monkeypatch.setattr(model_manager, "_save_metadata", fail_save)
    events = [event async for event in model_manager.pull_model("vllm-sr/d3-lite", "decision")]
    assert events[-1]["status"] == "error"
    assert not (settings.models_dir / "d3-lite").exists()
    assert (existing / "weights").read_text() == "preserved"


@pytest.mark.asyncio
async def test_d3_with_corrupted_output_is_not_registered(ollama_source, monkeypatch):
    mock_model_info(monkeypatch)

    async def worker(*args, **kwargs):
        artifact = Path(args[-1])
        create_d3_artifact(artifact)
        (artifact / "readout.safetensors").write_bytes(b"corrupted")
        return fake_worker()

    monkeypatch.setattr("mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", worker)
    events = [event async for event in model_manager.pull_model(D3_REPO, "decision")]
    assert events[-1]["status"] == "error"
    assert "checksum" in events[-1]["message"]
    assert "d3-lite" not in model_manager._metadata
    assert not (settings.models_dir / "d3-lite").exists()
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


@pytest.mark.asyncio
async def test_d3_preparation_uses_license_from_the_resolved_revision(ollama_source, monkeypatch):
    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.HfApi.model_info",
        lambda *args, **kwargs: SimpleNamespace(
            sha=D3_REVISION, siblings=[], card_data=SimpleNamespace(license="apache-2.0")
        ),
    )

    async def worker(*args, **kwargs):
        assert kwargs["env"]["MLX_SERVE_PREPARE_LICENSE"] == "apache-2.0"
        create_d3_artifact(Path(args[-1]))
        return fake_worker()

    monkeypatch.setattr("mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", worker)
    events = [event async for event in model_manager.pull_model(D3_REPO, "decision")]
    assert events[-1]["status"] == "success"


def test_d3_manifest_rechecks_identity_and_all_files(tmp_path):
    artifact = tmp_path / "artifact"
    create_d3_artifact(artifact)
    assert read_verified_manifest(artifact)["revision"] == D3_REVISION
    (artifact / "readout.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        read_verified_manifest(artifact)
    (artifact / "readout.safetensors").write_bytes(b"readout")
    manifest_file = artifact / MANIFEST
    manifest = json.loads(manifest_file.read_text())
    manifest["validation"]["published_file_hashes"] = False
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid"):
        read_verified_manifest(artifact)

    manifest["validation"]["published_file_hashes"] = True
    del manifest["files_sha256"]["model.safetensors"]
    (artifact / "model.safetensors").unlink()
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="backbone"):
        read_verified_manifest(artifact)


def test_prepared_d3_is_checked_again_before_loading(ollama_source, monkeypatch):
    artifact = settings.models_dir / "d3-lite"
    create_d3_artifact(artifact)
    model_manager._metadata["d3-lite"] = {
        "type": "decision", "hf_repo": D3_REPO, "revision": D3_REVISION,
    }
    model_manager._save_metadata()
    monkeypatch.setattr(model_manager, "_decision_cache", TTLLRUCache(maxsize=1, ttl=60))
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *args: None)
    monkeypatch.setattr(model_manager, "_memory_calibration_baseline", lambda: (0, None))
    monkeypatch.setattr(
        "mlx_serve.core.decision.D3DecisionModel",
        lambda path: pytest.fail("A modified checkpoint must never be loaded"),
    )
    (artifact / "readout.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        model_manager.get_decision_model("d3-lite")


def test_d3_worker_rejects_incorrect_published_hashes_before_loading(tmp_path, monkeypatch):
    def download(*, repo_id, revision, local_dir, allow_patterns):
        source = Path(local_dir)
        source.mkdir()
        hashes = {
            "tokenizer.json": "0" * 64,
            "tokenizer_config.json": "0" * 64,
            "readout.safetensors": "0" * 64,
            "model.safetensors": "0" * 64,
        }
        (source / "MODEL_MANIFEST.json").write_text(json.dumps({
            "repo_id": D3_REPO, "format_id": "d3-code-readout-v1",
            "identity": {"files_sha256": hashes},
        }))
        for name in hashes:
            (source / name).write_bytes(b"corrupted")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(ValueError, match="checksum"):
        prepare_d3(D3_REPO, D3_REVISION, tmp_path / "source", tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_d3_worker_rejects_extra_weight_files(tmp_path, monkeypatch):
    def download(*, repo_id, revision, local_dir, allow_patterns):
        source = Path(local_dir)
        source.mkdir()
        hashes = {name: "0" * 64 for name in (
            "tokenizer.json", "tokenizer_config.json", "readout.safetensors", "model.safetensors"
        )}
        (source / "MODEL_MANIFEST.json").write_text(json.dumps({
            "repo_id": D3_REPO, "format_id": "d3-code-readout-v1",
            "identity": {"files_sha256": hashes},
        }))
        for name in (*hashes, "unexpected.safetensors"):
            (source / name).write_bytes(b"unexpected")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(ValueError, match="unexpected or missing weight"):
        prepare_d3(D3_REPO, D3_REVISION, tmp_path / "source", tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


@pytest.mark.asyncio
async def test_cancelled_d3_download_stops_worker_before_cleaning_stage(
    ollama_source, monkeypatch
):
    mock_model_info(monkeypatch)
    started = asyncio.Event()
    stopped = []

    class Process:
        pid = os.getpid()
        returncode = None
        stdout = asyncio.StreamReader()

        def terminate(self):
            self.returncode = -15
            stopped.append(True)

        async def wait(self):
            return self.returncode

    async def worker(*args, **kwargs):
        started.set()
        return Process()

    monkeypatch.setattr("mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", worker)

    async def install():
        return [event async for event in model_manager.pull_model(D3_REPO, "decision")]

    pending = asyncio.create_task(install())
    await asyncio.wait_for(started.wait(), timeout=3)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert stopped
    assert not (settings.models_dir / "d3-lite").exists()
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


@pytest.mark.asyncio
async def test_import_serializes_registration_and_preserves_newer_metadata(
    ollama_source, monkeypatch
):
    settings.models_dir.mkdir()
    lock_path = settings.models_dir.parent / ".decision-metadata.lock"
    started = threading.Event()
    real_link = os.link

    def signal_link(source, target):
        result = real_link(source, target)
        started.set()
        return result

    monkeypatch.setattr("mlx_serve.core.ollama_import.os.link", signal_link)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        task = asyncio.create_task(
            asyncio.to_thread(import_ollama_decision, "d3-lite:mlx", "d3-lite", ollama_source)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await asyncio.sleep(0.05)
            assert not task.done()
            model_manager._metadata_path.write_text(json.dumps({"d1": {"type": "decision"}}))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        destination = await task
    assert destination.exists()
    metadata = json.loads(model_manager._metadata_path.read_text())
    assert metadata["d1"] == {"type": "decision"}
    assert metadata["d3-lite"]["source"] == "ollama:d3-lite:mlx"


def test_import_preserves_destination_created_during_staging(ollama_source, monkeypatch):
    destination = settings.models_dir / "d3-lite"
    real_link = os.link

    def concurrent_install(source, target):
        if not destination.exists():
            destination.mkdir()
            (destination / "weights").write_text("existing")
        return real_link(source, target)

    monkeypatch.setattr("mlx_serve.core.ollama_import.os.link", concurrent_install)
    with pytest.raises(FileExistsError):
        import_ollama_decision("d3-lite:mlx", "d3-lite", ollama_source)
    assert (destination / "weights").read_text() == "existing"
    assert [path.name for path in settings.models_dir.iterdir()] == ["d3-lite"]
