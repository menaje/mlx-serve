"""Regression tests for d1 validation, registration and the decision contract."""

import asyncio
import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_serve.config import settings
from mlx_serve.core.d1_decision import D1DecisionModel, DecisionInputError
from mlx_serve.core.decision_prepare import prepare_d1, read_verified_manifest
from mlx_serve.core.decision_prepare_worker import (
    MANIFEST,
    MLX_REPO,
    SOURCE_REPO,
    compare_predictions,
    file_hash,
    validate_checkpoint,
)
from mlx_serve.core.model_manager import TTLLRUCache, model_manager

REVISION = "0123456789abcdef0123456789abcdef01234567"


def create_artifact(path: Path, repo: str = SOURCE_REPO):
    path.mkdir(parents=True)
    for name, data in {
        "config.json": b'{"model_type":"d1_omni"}',
        "tokenizer.json": b"{}",
        "LICENSE": b"license",
        "model.safetensors": b"weights",
    }.items():
        (path / name).write_bytes(data)
    files = {p.name: file_hash(p) for p in path.iterdir()}
    manifest = {
        "schema_version": 1,
        "source_repo": repo,
        "revision": REVISION,
        "architecture": "d1_omni",
        "precision": "float16",
        "files_sha256": files,
        "validation": {"official_predict": True},
    }
    (path / MANIFEST).write_text(json.dumps(manifest))


def test_manifest_detects_changed_and_unexpected_files(tmp_path):
    create_artifact(tmp_path / "artifact")
    path = tmp_path / "artifact"
    assert read_verified_manifest(path)["revision"] == REVISION
    (path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_verified_manifest(path)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "extra.safetensors").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        read_verified_manifest(path)


def test_checkpoint_checks_actual_head_precision_and_required_files(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    for filename in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        (tmp_path / filename).write_text("{}")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "d1_omni", "temperatures": {"choice:2": 1.0}})
    )
    weights = tmp_path / "model.safetensors"
    save_file(
        {"head.head.layers.0.attention.query_proj.weight": np.ones((2, 2), dtype="float16")},
        weights,
    )
    assert validate_checkpoint(tmp_path, converted=True)["model_type"] == "d1_omni"
    save_file(
        {"head.head.layers.0.attention.query_proj.weight": np.ones((2, 2), dtype="float32")},
        weights,
    )
    with pytest.raises(ValueError, match="tensor types"):
        validate_checkpoint(tmp_path, converted=True)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        validate_checkpoint(tmp_path, converted=True)


def test_probability_comparison_preserves_option_ids_and_tolerance():
    original = {
        "answers": {"x": {"type": "choice", "value": "b", "probabilities": {"a": 0.1, "b": 0.9}}}
    }
    converted = {
        "answers": {
            "x": {"type": "choice", "value": "b", "probabilities": {"b": 0.899, "a": 0.101}}
        }
    }
    assert compare_predictions(original, converted) == pytest.approx(0.001)
    converted["answers"]["x"]["probabilities"] = {"b": 0.1, "a": 0.9}
    with pytest.raises(ValueError, match="probabilities"):
        compare_predictions(original, converted)


def test_rejects_d1_state_and_choice_truncation():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text)}

    model = object.__new__(D1DecisionModel)
    model.processor = Tokenizer()
    model.max_tokens = 128
    with pytest.raises(DecisionInputError, match="State"):
        model._check_input("long state " * 100, {"type": "bool", "instructions": "test"})
    with pytest.raises(DecisionInputError, match="choice"):
        model._check_input(
            "ok",
            {"type": "choice", "instructions": "test", "criteria": {"a": "x" * 200, "b": "ok"}},
        )


def test_decisions_returns_official_answers_and_revision(client, monkeypatch):
    class Model:
        def decide(self, state, questions):
            return {
                "answers": {
                    "selected": {
                        "type": "choice",
                        "value": "b",
                        "probabilities": {"a": 0.1, "b": 0.9},
                    },
                    "yes": {"type": "bool", "value": True, "probability": 0.8},
                    "score": {
                        "type": "score",
                        "value": 0.8,
                        "probabilities": {"0": 0.2, "1": 0.8},
                        "metadata": {"legend": {"0": "bad", "1": "good"}},
                    },
                },
                "usage": {"input_tokens": 12, "output_tokens": 0},
            }

    monkeypatch.setattr(model_manager, "get_decision_model", lambda name: Model())
    monkeypatch.setattr(model_manager, "_metadata", {"d1-omni-600M": {"revision": REVISION}})
    response = client.post(
        "/v1/decisions",
        json={
            "model": "d1-omni",
            "state": {"text": "test"},
            "questions": {
                "selected": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                "yes": {"type": "noul"},
                "score": {"type": "score", "criteria": ["bad", "good"]},
            },
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["revision"] == REVISION and result["model"] == "d1-omni-600M"
    assert result["answers"]["selected"]["value"] == "b"
    assert result["answers"]["yes"]["value"] is True
    assert result["answers"]["score"]["metadata"]["legend"]["1"] == "good"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 0}


def test_decisions_reports_truncation_as_422(client, monkeypatch):
    class Model:
        def decide(self, *args):
            raise DecisionInputError("State exceeds the input limit")

    monkeypatch.setattr(model_manager, "get_decision_model", lambda name: Model())
    response = client.post(
        "/v1/decisions",
        json={
            "model": "d1-omni",
            "state": "test",
            "questions": {"q": {"type": "bool"}},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "decision_input_too_long"


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    settings.models_dir.mkdir()
    monkeypatch.setattr(model_manager, "_metadata_path", tmp_path / "metadata.json")
    monkeypatch.setattr(model_manager, "_metadata", {})
    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.HfApi.model_info",
        lambda *a, **k: SimpleNamespace(sha=REVISION, siblings=[]),
    )
    return tmp_path


@pytest.mark.parametrize("alias", ["d1-omni", "D1-OMNI", "d1-omni-600m", "d1-omni-600M"])
def test_api_pull_alias_registers_a_callable_canonical_model(
    preparation, monkeypatch, client, alias
):
    async def fake_spawn(*args, **kwargs):
        create_artifact(Path(args[-1]))
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    class FakeDecisionModel:
        def __init__(self, model_dir):
            assert model_dir == settings.models_dir / "d1-omni-600M"

        def decide(self, state, questions):
            return {
                "answers": {"q": {"type": "bool", "value": True, "probability": 0.9}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )
    monkeypatch.setattr("mlx_serve.core.d1_decision.D1DecisionModel", FakeDecisionModel)
    monkeypatch.setattr(model_manager, "_decision_cache", TTLLRUCache(maxsize=1, ttl=60))
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(model_manager, "_memory_calibration_baseline", lambda: (0, 0))
    monkeypatch.setattr(model_manager, "_calibrate_loaded_model_estimate", lambda *a, **k: None)

    response = client.post(
        "/api/pull",
        json={"name": SOURCE_REPO, "type": "decision", "model_name": alias, "revision": REVISION},
    )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["status"] == "success"
    assert events[-1]["name"] == "d1-omni-600M"
    assert model_manager.is_model_installed(alias)
    assert model_manager.get_model_info(alias).name == "d1-omni-600M"
    assert set(model_manager._metadata) == {"d1-omni-600M"}

    decision = client.post(
        "/v1/decisions",
        json={"model": alias, "state": "test", "questions": {"q": {"type": "bool"}}},
    )
    assert decision.status_code == 200
    assert decision.json()["model"] == "d1-omni-600M"
    assert decision.json()["answers"]["q"]["value"] is True


@pytest.mark.parametrize(
    ("repo", "alias", "canonical"),
    [
        (SOURCE_REPO, "d1-omni", "d1-omni-600M"),
        (MLX_REPO, "D1-OMNI-600M-MLX-FP16", "d1-omni-600M-MLX-fp16"),
    ],
)
@pytest.mark.asyncio
async def test_direct_preparation_normalizes_aliases(
    preparation, monkeypatch, repo, alias, canonical
):
    async def fake_spawn(*args, **kwargs):
        create_artifact(Path(args[-1]), repo)
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )
    events = [event async for event in prepare_d1(model_manager, repo, alias, REVISION)]
    assert events[-1]["status"] == "success"
    assert events[-1]["name"] == canonical
    assert model_manager.is_model_installed(alias)
    assert (settings.models_dir / canonical).is_dir()


@pytest.mark.parametrize(
    ("repo", "name"),
    [
        (SOURCE_REPO, "d3-lite"),
        (SOURCE_REPO, "qwen-embedding"),
        (SOURCE_REPO, "d1-omni-600m-mlx-fp16"),
        (MLX_REPO, "d1-omni"),
    ],
)
@pytest.mark.asyncio
async def test_preparation_rejects_aliases_reserved_for_other_sources(preparation, repo, name):
    direct = [event async for event in prepare_d1(model_manager, repo, name, REVISION)]
    through_manager = [
        event async for event in model_manager.pull_model(repo, "decision", name, revision=REVISION)
    ]
    for events in (direct, through_manager):
        assert events[-1]["status"] == "error"
        assert "reserved for" in events[-1]["message"]
    assert not list(settings.models_dir.iterdir())
    assert not model_manager._metadata


@pytest.mark.asyncio
async def test_alias_does_not_replace_a_conflicting_canonical_installation(preparation):
    destination = settings.models_dir / "d1-omni-600M"
    create_artifact(destination)
    model_manager._metadata["d1-omni-600M"] = {
        "type": "decision",
        "hf_repo": SOURCE_REPO,
        "revision": "another-revision",
    }
    model_manager._save_metadata()
    previous_metadata = model_manager._metadata_path.read_bytes()
    results = [
        event
        async for event in model_manager.pull_model(
            SOURCE_REPO, "decision", "d1-omni", revision=REVISION
        )
    ]
    assert results[-1]["status"] == "error"
    assert "does not match" in results[-1]["message"]
    assert model_manager._metadata_path.read_bytes() == previous_metadata
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert not (settings.models_dir / "d1-omni").exists()


@pytest.mark.asyncio
async def test_preparation_registers_once_and_reuses_verified_model(preparation, monkeypatch):
    calls = 0

    async def fake_spawn(*args, **kwargs):
        nonlocal calls
        calls += 1
        create_artifact(Path(args[-1]))
        reader = asyncio.StreamReader()
        reader.feed_data(b'MLX_PREP_STATUS={"status":"ready"}\n')
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )
    results = [event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)]
    assert results[-1]["status"] == "success"
    assert model_manager._metadata["d1-test"]["revision"] == REVISION
    assert model_manager._metadata["d1-test"]["size"] > 0
    reused = [event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)]
    assert reused[-1]["reused"] is True
    assert calls == 1
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


@pytest.mark.asyncio
async def test_preparation_failure_preserves_existing_model_and_metadata(preparation, monkeypatch):
    destination = settings.models_dir / "d1-test"
    destination.mkdir()
    (destination / "user-data").write_text("untouched")
    model_manager._metadata["d1-test"] = {"type": "decision", "revision": "old"}
    model_manager._save_metadata()
    before = model_manager._metadata_path.read_bytes()
    result = [event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)]
    assert result[-1]["status"] == "error"
    assert (destination / "user-data").read_text() == "untouched"
    assert model_manager._metadata_path.read_bytes() == before


@pytest.mark.asyncio
async def test_registration_failure_removes_unpublished_artifact(preparation, monkeypatch):
    async def fake_spawn(*args, **kwargs):
        create_artifact(Path(args[-1]))
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )
    monkeypatch.setattr(
        model_manager, "_save_metadata", lambda: (_ for _ in ()).throw(OSError("disk"))
    )
    result = [event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)]
    assert result[-1]["status"] == "error"
    assert not (settings.models_dir / "d1-test").exists()
    assert "d1-test" not in model_manager._metadata
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


def test_timeout_returns_while_inference_finishes(client, monkeypatch):
    class SlowModel:
        def decide(self, *args):
            time.sleep(0.2)
            return {"answers": {}, "usage": {}}

    monkeypatch.setattr(model_manager, "get_decision_model", lambda name: SlowModel())
    monkeypatch.setattr(settings, "decision_execution_timeout_seconds", 0.02)
    response = client.post(
        "/v1/decisions",
        json={
            "model": "d1-omni",
            "state": "test",
            "questions": {"q": {"type": "bool"}},
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"]["error"]["code"] == "decision_timeout"


@pytest.mark.asyncio
async def test_unsupported_name_cannot_masquerade_as_mlx(preparation):
    result = [
        event async for event in model_manager.pull_model("unknown/not-really-mlx-4bit", "decision")
    ]
    assert result[-1]["status"] == "error"
    assert "Unsupported" in result[-1]["message"]
    assert not (settings.models_dir / "not-really-mlx-4bit").exists()


@pytest.mark.asyncio
async def test_concurrent_same_revision_prepares_once(preparation, monkeypatch):
    calls = 0

    async def fake_spawn(*args, **kwargs):
        nonlocal calls
        calls += 1
        create_artifact(Path(args[-1]))
        await asyncio.sleep(0.05)
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )

    async def install():
        return [
            event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)
        ]

    first, second = await asyncio.gather(install(), install())
    assert calls == 1
    assert first[-1]["status"] == second[-1]["status"] == "success"
    assert sorted([first[-1]["reused"], second[-1]["reused"]]) == [False, True]
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


@pytest.mark.asyncio
async def test_cancelled_preparation_stops_worker_and_removes_stage(preparation, monkeypatch):
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

    async def fake_spawn(*args, **kwargs):
        started.set()
        return Process()

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )

    async def install():
        return [
            event async for event in prepare_d1(model_manager, SOURCE_REPO, "d1-test", REVISION)
        ]

    task = asyncio.create_task(install())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped
    assert not (settings.models_dir / "d1-test").exists()
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


@pytest.mark.asyncio
async def test_prepare_can_retain_its_own_original(preparation, monkeypatch):
    async def fake_spawn(*args, **kwargs):
        source, artifact = Path(args[-2]), Path(args[-1])
        source.mkdir(parents=True)
        (source / "original-weights").write_bytes(b"source")
        create_artifact(artifact)
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )
    results = [
        event
        async for event in prepare_d1(
            model_manager, SOURCE_REPO, "d1-test", REVISION, keep_original=True
        )
    ]
    assert results[-1]["status"] == "success"
    original = Path(model_manager._metadata["d1-test"]["original_path"])
    assert (original / "original-weights").read_bytes() == b"source"
    assert not list(settings.models_dir.glob(".decision-prepare-*"))


def test_pull_api_passes_pinned_revision(client, monkeypatch):
    captured = {}

    async def pull(**kwargs):
        captured.update(kwargs)
        yield {"status": "success", "name": "d1-test"}

    monkeypatch.setattr(model_manager, "pull_model", pull)
    response = client.post(
        "/api/pull",
        json={
            "name": SOURCE_REPO,
            "type": "decision",
            "revision": REVISION,
        },
    )
    assert response.status_code == 200
    assert json.loads(response.text)["status"] == "success"
    assert captured["revision"] == REVISION


def test_auto_download_timeout_cancels_preparation(monkeypatch):
    cleaned = []

    async def pull(*args, **kwargs):
        try:
            yield {"status": "downloading"}
            await asyncio.sleep(10)
        finally:
            cleaned.append(True)

    monkeypatch.setattr(settings, "auto_download_timeout", 0.05)
    monkeypatch.setattr(model_manager, "pull_model", pull)
    started = time.monotonic()
    assert model_manager._auto_download_model(SOURCE_REPO, "decision", "d1-test") is False
    assert time.monotonic() - started < 1
    assert cleaned


@pytest.mark.asyncio
async def test_same_revision_under_different_names_reuses_without_reconversion(
    preparation, monkeypatch
):
    calls = 0

    async def fake_spawn(*args, **kwargs):
        nonlocal calls
        calls += 1
        create_artifact(Path(args[-1]))
        await asyncio.sleep(0.05)
        reader = asyncio.StreamReader()
        reader.feed_eof()
        return SimpleNamespace(
            stdout=reader, pid=os.getpid(), returncode=0, wait=lambda: asyncio.sleep(0, result=0)
        )

    monkeypatch.setattr(
        "mlx_serve.core.decision_prepare.asyncio.create_subprocess_exec", fake_spawn
    )

    async def install(name):
        return [event async for event in prepare_d1(model_manager, SOURCE_REPO, name, REVISION)]

    first, second = await asyncio.gather(install("first"), install("second"))
    assert first[-1]["status"] == second[-1]["status"] == "success"
    assert calls == 1
    assert sorted([first[-1]["reused"], second[-1]["reused"]]) == [False, True]
    original = settings.models_dir / "first" / "model.safetensors"
    copy = settings.models_dir / "second" / "model.safetensors"
    assert original.stat().st_ino != copy.stat().st_ino
    copy.write_bytes(b"changed")
    assert original.read_bytes() == b"weights"


def test_gateway_unloads_the_isolated_decision_worker(client, monkeypatch):
    from mlx_serve.routers.retrieval_proxy import WorkerProxyResponse

    client.app.state.retrieval_worker_urls = {"decision": "http://worker"}
    seen = []
    worker_cache = {
        "decision_models": {"count": 0, "models": []},
        "mlx_memory": {"available": True, "active_bytes": 0},
        "active_inference": [],
    }

    def worker_request(url, method, body, headers):
        seen.append((url, method, json.loads(body)))
        payload = {
            "unloaded": [{"name": "d1-omni-600M", "type": "decision"}],
            "skipped_active": [],
            "cache": worker_cache,
        }
        return WorkerProxyResponse(200, json.dumps(payload).encode(), {})

    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", worker_request)
    response = client.post("/v1/models/d1-omni/unload?type=decision")
    assert response.status_code == 200
    assert response.json()["unloaded"] == [{"name": "d1-omni-600M", "type": "decision"}]
    assert response.json()["cache"]["decision_models"]["count"] == 0
    assert seen == [("http://worker/internal/decision/unload", "POST", {"name": "d1-omni"})]


@pytest.mark.asyncio
async def test_timed_out_inference_keeps_model_reserved_until_thread_finishes(monkeypatch):
    import threading

    import httpx

    from mlx_serve.core.model_manager import TTLLRUCache
    from mlx_serve.server import create_app

    release = threading.Event()

    class SlowModel:
        def decide(self, *args):
            release.wait(timeout=3)
            return {"answers": {}, "usage": {}}

    cache = TTLLRUCache(maxsize=1, ttl=60)
    cache.set("d1-omni-600M", SlowModel())
    monkeypatch.setattr(model_manager, "_decision_cache", cache)
    monkeypatch.setattr(model_manager, "get_decision_model", lambda name: cache.get("d1-omni-600M"))
    monkeypatch.setattr(settings, "decision_execution_timeout_seconds", 0.02)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    transport = httpx.ASGITransport(app=create_app())
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/decisions",
                json={
                    "model": "d1-omni",
                    "state": "test",
                    "questions": {"q": {"type": "bool"}},
                },
            )
            assert response.status_code == 504
            blocked = await client.post("/v1/models/d1-omni/unload?type=decision")
            assert blocked.status_code == 409
            release.set()
            for _ in range(50):
                if not model_manager._active_inference_keys():
                    break
                await asyncio.sleep(0.01)
            completed = await client.post("/v1/models/d1-omni/unload?type=decision")
            assert completed.status_code == 200
            assert completed.json()["unloaded"] == [{"name": "d1-omni-600M", "type": "decision"}]
    finally:
        release.set()


def test_worker_refreshes_reinstalled_revision_and_invalidates_external_deletion(
    preparation, monkeypatch
):
    directory = settings.models_dir / "d1-omni-600M"
    create_artifact(directory)
    installed = {"type": "decision", "revision": REVISION, "hf_repo": SOURCE_REPO}
    model_manager._metadata_path.write_text(json.dumps({directory.name: installed}))
    monkeypatch.setattr(
        model_manager,
        "_metadata",
        {directory.name: {**installed, "revision": "old-revision"}},
    )
    monkeypatch.setattr(model_manager, "_decision_cache", TTLLRUCache(maxsize=1, ttl=60))
    monkeypatch.setattr(model_manager, "_decision_cache_identity", {})
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(model_manager, "_memory_calibration_baseline", lambda: (0, 0))
    monkeypatch.setattr(model_manager, "_calibrate_loaded_model_estimate", lambda *a, **k: None)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(settings, "auto_download", False)

    class FakeDecisionModel:
        def __init__(self, model_dir):
            self.model_dir = model_dir

    monkeypatch.setattr("mlx_serve.core.d1_decision.D1DecisionModel", FakeDecisionModel)
    first = model_manager.get_decision_model("d1-omni")
    assert model_manager._metadata[directory.name]["revision"] == REVISION
    assert model_manager.get_decision_model("d1-omni") is first

    # Simulate a CLI replacing the same name without an intervening worker request.
    shutil.rmtree(directory)
    create_artifact(directory)
    new_revision = "fedcba9876543210fedcba9876543210fedcba98"
    manifest = json.loads((directory / MANIFEST).read_text())
    manifest["revision"] = new_revision
    (directory / MANIFEST).write_text(json.dumps(manifest))
    model_manager._metadata_path.write_text(
        json.dumps({directory.name: {**installed, "revision": new_revision}})
    )
    replacement = model_manager.get_decision_model("d1-omni")
    assert replacement is not first
    assert model_manager._metadata[directory.name]["revision"] == new_revision

    shutil.rmtree(directory)
    model_manager._metadata_path.write_text("{}")
    with pytest.raises(FileNotFoundError):
        model_manager.get_decision_model("d1-omni")
    assert model_manager._decision_cache.get(directory.name) is None


@pytest.mark.asyncio
async def test_worker_delete_blocks_loading_then_removes_cached_model(preparation, monkeypatch):
    import httpx

    from mlx_serve.core.runtime_topology import RETRIEVAL_WORKER_KIND_ENV, SERVER_ROLE_ENV
    from mlx_serve.server import create_app

    name = "d1-omni-600M"
    create_artifact(settings.models_dir / name)
    model_manager._metadata = {
        name: {"type": "decision", "revision": REVISION, "hf_repo": SOURCE_REPO},
        "other": {"type": "embedding"},
    }
    model_manager._save_metadata()
    cache = TTLLRUCache(maxsize=1, ttl=60)
    monkeypatch.setattr(model_manager, "_decision_cache", cache)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(RETRIEVAL_WORKER_KIND_ENV, "decision")
    started = threading.Event()
    release = threading.Event()

    class SlowModel:
        def decide(self, *args):
            return {"answers": {}, "usage": {}}

    def load_model(model_name):
        started.set()
        release.wait(timeout=5)
        model = SlowModel()
        cache.set(name, model)
        return model

    monkeypatch.setattr(model_manager, "get_decision_model", load_model)
    transport = httpx.ASGITransport(app=create_app())
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            running = asyncio.create_task(
                client.post(
                    "/v1/decisions",
                    json={
                        "model": "d1-omni",
                        "state": "test",
                        "questions": {"q": {"type": "bool"}},
                    },
                )
            )
            assert await asyncio.to_thread(started.wait, 3)
            blocked = await client.post("/internal/decision/delete", json={"name": "d1-omni"})
            assert blocked.status_code == 409
            assert (settings.models_dir / name).exists()
            release.set()
            assert (await running).status_code == 200
            deleted = await client.post("/internal/decision/delete", json={"name": "d1-omni"})
            assert deleted.json() == {"deleted": True}
            assert not (settings.models_dir / name).exists()
            assert cache.get(name) is None
            assert json.loads(model_manager._metadata_path.read_text()) == {
                "other": {"type": "embedding"}
            }
    finally:
        release.set()


def test_gateway_deletes_in_worker_and_handles_worker_unavailable(preparation, monkeypatch, client):
    import urllib.error

    from mlx_serve.routers.retrieval_proxy import WorkerProxyResponse

    name = "d1-omni-600M"
    create_artifact(settings.models_dir / name)
    model_manager._metadata[name] = {
        "type": "decision",
        "revision": REVISION,
        "hf_repo": SOURCE_REPO,
    }
    model_manager._save_metadata()
    client.app.state.retrieval_worker_urls = {"decision": "http://worker"}
    calls = []

    def unavailable(url, method, body, headers):
        raise urllib.error.URLError("worker down")

    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", unavailable)
    response = client.request("DELETE", "/api/delete", json={"name": "d1-omni"})
    assert response.status_code == 503
    assert (settings.models_dir / name).exists()

    def worker_request(url, method, body, headers):
        calls.append((url, method, json.loads(body)))
        model_manager.delete_model("d1-omni")
        return WorkerProxyResponse(200, b'{"deleted": true}', {})

    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", worker_request)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    response = client.request("DELETE", "/api/delete", json={"name": "d1-omni"})
    assert response.status_code == 200
    assert calls == [("http://worker/internal/decision/delete", "POST", {"name": "d1-omni"})]
    assert not (settings.models_dir / name).exists()
