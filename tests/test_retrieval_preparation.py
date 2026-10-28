"""Validated retrieval profiles use immutable files and the persistent job path."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_serve.core.decision_prepare import read_verified_retrieval_manifest
from mlx_serve.core.decision_prepare_worker import file_hash
from mlx_serve.core.model_manager import model_manager
from mlx_serve.core.retrieval_prepare_worker import (
    EMBEDDING_REPO,
    MANIFEST,
    _git_blob_hash,
    verify_published_files,
)

REVISION = "a" * 40


def make_artifact(path: Path):
    path.mkdir()
    for name, data in {
        "config.json": b'{}', "tokenizer.json": b'{}', "model.safetensors": b'weights',
    }.items():
        (path / name).write_bytes(data)
    manifest = {
        "schema_version": 1, "source_repo": EMBEDDING_REPO, "type": "embedding",
        "revision": REVISION, "architecture": "bert", "precision": "bfloat16",
        "validation": {"mlx_predict": True, "published_file_hashes": True},
        "files_sha256": {p.name: file_hash(p) for p in path.iterdir()},
    }
    (path / MANIFEST).write_text(json.dumps(manifest))
    return manifest


def test_published_git_and_lfs_hashes_and_complete_file_set(tmp_path):
    for name, contents in {
        "config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"weights",
    }.items():
        (tmp_path / name).write_bytes(contents)
    siblings = []
    for path in tmp_path.iterdir():
        lfs = SimpleNamespace(sha256=file_hash(path)) if path.suffix == ".safetensors" else None
        siblings.append(SimpleNamespace(rfilename=path.name, size=path.stat().st_size,
                                        blob_id=_git_blob_hash(path), lfs=lfs))
    assert set(verify_published_files(tmp_path, siblings)) == {
        "config.json", "tokenizer.json", "model.safetensors"
    }
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_published_files(tmp_path, siblings)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="required"):
        verify_published_files(tmp_path, siblings)
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    siblings.append(SimpleNamespace(rfilename="extra.json", size=2, blob_id="a" * 40, lfs=None))
    with pytest.raises(ValueError, match="published revision"):
        verify_published_files(tmp_path, siblings)


def test_manifest_rechecks_files_and_registered_identity(tmp_path, monkeypatch):
    path = tmp_path / "model"
    make_artifact(path)
    assert read_verified_retrieval_manifest(path)["revision"] == REVISION
    metadata = {"embedding": {"preparation_format": "retrieval-v1", "type": "embedding",
                              "hf_repo": EMBEDDING_REPO, "revision": REVISION}}
    monkeypatch.setattr(model_manager, "_metadata", metadata)
    monkeypatch.setattr(model_manager, "_load_metadata", lambda: None)
    model_manager._verify_retrieval_preparation("embedding", "embedding", path)
    metadata["embedding"]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="metadata"):
        model_manager._verify_retrieval_preparation("embedding", "embedding", path)
    metadata["embedding"]["revision"] = REVISION
    (path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        model_manager._verify_retrieval_preparation("embedding", "embedding", path)


def test_retrieval_manifest_rejects_malformed_data_and_links(tmp_path):
    path = tmp_path / "model"
    manifest = make_artifact(path)
    for invalid in ([], {**manifest, "source_repo": []}):
        (path / MANIFEST).write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="manifest is invalid"):
            read_verified_retrieval_manifest(path)
    (path / MANIFEST).write_text(json.dumps(manifest))
    (path / "extra-link").symlink_to(path / "config.json")
    with pytest.raises(ValueError, match="Linked"):
        read_verified_retrieval_manifest(path)


def test_recovery_finds_both_worker_types_for_the_correct_job(monkeypatch):
    import mlx_serve.core.managed_jobs as jobs_module

    result = SimpleNamespace(stdout="\n".join([
        "101 python -m mlx_serve.core.decision_prepare_worker /tmp/.decision-prepare-abc-one",
        "102 python -m mlx_serve.core.retrieval_prepare_worker /tmp/.decision-prepare-abc-two",
        "103 python -m mlx_serve.core.retrieval_prepare_worker /tmp/.decision-prepare-other-two",
        "104 python unrelated /tmp/.decision-prepare-abc-three",
    ]))
    monkeypatch.setattr(jobs_module.subprocess, "run", lambda *args, **kwargs: result)
    assert jobs_module._job_worker_pids("abc") == [101, 102]


@pytest.mark.asyncio
async def test_retrieval_job_loads_using_its_own_type(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as jobs_module

    models = tmp_path / "models"
    models.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import json, sys\njson.loads(sys.stdin.readline())\n"
        "print(json.dumps({'status':'success','name':'embed'}), flush=True)\n"
    )
    spawn = asyncio.create_subprocess_exec

    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, str(runner), **kwargs)

    calls = []

    async def load(name, kind):
        calls.append((name, kind))

    monkeypatch.setattr(jobs_module.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(jobs_module, "_job_worker_pids", lambda _: [])
    jobs = jobs_module.JobManager(tmp_path / "jobs", load_model=load)
    job = jobs.start({"repo": EMBEDDING_REPO, "models_dir": str(models), "load_after": True})
    await jobs._tasks[job["id"]]
    assert jobs.get(job["id"])["result"]["auto_load"] == "succeeded"
    assert calls == [("embed", "embedding")]
    await jobs.close()
