"""Management and lifetime guarantees for isolated generation workers."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mlx_serve.config import settings
from mlx_serve.core.generation_workers import (
    GenerationWorkerProcess,
    GenerationWorkerSupervisor,
    acquire_generation_worker,
    await_worker_operation,
)
from mlx_serve.core.model_manager import model_manager
from mlx_serve.server import create_app


class FakeProcess:
    pid = 12345

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def fake_worker(kind, model_name=None, port=19000):
    return GenerationWorkerProcess(
        kind=kind, host="127.0.0.1", port=port,
        process=FakeProcess(), model_name=model_name,
    )


def test_simultaneous_worker_starts_share_one_process_and_reaper_waits(monkeypatch):
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    monkeypatch.setattr(settings, "generation_worker_idle_timeout_seconds", 5.0)
    supervisor = GenerationWorkerSupervisor()
    starts = []

    def start(kind, model_name=None):
        starts.append(model_name)
        time.sleep(0.05)
        return fake_worker(kind, model_name)

    monkeypatch.setattr(supervisor, "_start_worker", start)
    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(lambda _: supervisor.acquire_worker("llm", "model-a"), range(8)))

    worker = supervisor._workers["llm:model-a"]
    assert starts == ["model-a"]
    assert worker.active_leases == 8
    assert supervisor.reap_idle_workers(now=time.monotonic() + 30) == []
    for lease in leases:
        lease.release()
        lease.release()
    assert worker.active_leases == 0
    assert supervisor.reap_idle_workers(now=time.monotonic() + 30) == ["llm:model-a"]
    assert worker.process.returncode == 0


def test_supervisor_stops_a_worker_whose_start_finishes_during_shutdown(monkeypatch):
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    supervisor = GenerationWorkerSupervisor()
    begun, finish = threading.Event(), threading.Event()
    worker = fake_worker("llm", "model-a")

    def start(*args, **kwargs):
        begun.set()
        assert finish.wait(3)
        return worker

    monkeypatch.setattr(supervisor, "_start_worker", start)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.acquire_worker, "llm", "model-a")
        assert begun.wait(3)
        supervisor.stop()
        finish.set()
        with pytest.raises(RuntimeError, match="stopping"):
            future.result(timeout=3)
    assert worker.process.returncode == 0
    assert supervisor._workers == {}


@pytest.mark.asyncio
async def test_cancelled_operation_keeps_worker_until_underlying_work_finishes(monkeypatch):
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    supervisor = GenerationWorkerSupervisor()
    monkeypatch.setattr(supervisor, "_start_worker", fake_worker)
    lease = await acquire_generation_worker(supervisor, "llm", "model-a")
    worker = supervisor._workers["llm:model-a"]
    begun, finish = asyncio.Event(), asyncio.Event()

    async def operation():
        begun.set()
        await finish.wait()
        return True

    task = asyncio.create_task(await_worker_operation(operation(), lease))
    await begun.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.active_leases == 1
    finish.set()
    for _ in range(20):
        if worker.active_leases == 0:
            break
        await asyncio.sleep(0.01)
    assert worker.active_leases == 0


@pytest.mark.asyncio
async def test_cancelled_start_releases_the_eventual_worker(monkeypatch):
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    supervisor = GenerationWorkerSupervisor()
    begun, finish = threading.Event(), threading.Event()

    def start(kind, model_name=None):
        begun.set()
        assert finish.wait(3)
        return fake_worker(kind, model_name)

    monkeypatch.setattr(supervisor, "_start_worker", start)
    task = asyncio.create_task(acquire_generation_worker(supervisor, "llm", "model-a"))
    assert await asyncio.to_thread(begun.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finish.set()
    for _ in range(100):
        worker = supervisor._workers.get("llm:model-a")
        if worker is not None and worker.active_leases == 0:
            break
        await asyncio.sleep(0.01)
    assert worker is not None and worker.active_leases == 0


def test_management_routes_to_the_model_worker_and_reports_its_actual_state(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    models = tmp_path / "models"
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(model_manager, "_metadata_path", tmp_path / "metadata.json")
    for name in ("model-a", "model-b", "model-c"):
        (models / name).mkdir(parents=True)
    (tmp_path / "metadata.json").write_text(
        '{"model-a":{"type":"llm"}, "model-b":{"type":"vlm"}, '
        '"model-c":{"type":"llm"}}'
    )
    supervisor = GenerationWorkerSupervisor()
    ports = {"model-a": 19001, "model-b": 19002, "model-c": 19003}
    monkeypatch.setattr(
        supervisor, "_start_worker",
        lambda kind, model_name=None: fake_worker(kind, model_name, ports[model_name]),
    )
    calls = []

    async def worker_request(url, path, payload=None):
        calls.append((url, path, payload))
        if path != "/internal/management/state":
            worker = supervisor._workers[f'{payload["type"]}:{payload["name"]}']
            assert worker.active_leases == 1
            if payload["name"] == "model-b" and path.endswith("unload"):
                raise HTTPException(status_code=409, detail="Model is processing a request")
            return {"loaded": True} if path.endswith("load") else {"unloaded": []}
        if url.endswith("19003"):
            raise HTTPException(status_code=503)
        name = "model-a" if url.endswith("19001") else "model-b"
        kind = "llm" if name == "model-a" else "vlm"
        return {
            "cache": {f"{kind}_models": {"models": [name]}},
            "requests": {f"{kind}:{name}": {"active": int(name == "model-a"), "queued": 0}},
        }

    monkeypatch.setattr("mlx_serve.routers.management._worker_request", worker_request)
    app = create_app()
    app.state.generation_worker_supervisor = supervisor
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    for name, kind in (("model-a", "llm"), ("model-b", "vlm"), ("model-c", "llm")):
        response = client.post("/management/models/load", json={"name": name, "type": kind})
        assert response.status_code == 200
    blocked = client.post("/management/models/unload", json={"name": "model-b", "type": "vlm"})
    assert blocked.status_code == 409
    assert supervisor._workers["vlm:model-b"].active_leases == 0
    listed = {item["name"]: item for item in client.get("/management/state").json()["models"]}
    assert (listed["model-a"]["load_state"], listed["model-a"]["request_state"]) == (
        "loaded", "running",
    )
    assert (listed["model-b"]["load_state"], listed["model-b"]["request_state"]) == (
        "loaded", "idle",
    )
    assert (listed["model-c"]["load_state"], listed["model-c"]["request_state"]) == (
        "unknown", "unknown",
    )
    assert any(url.endswith("19002") and path.endswith("unload") for url, path, _ in calls)
    assert client.post(
        "/management/models/load", json={"name": "model-a", "type": "vlm"}
    ).status_code == 400
    assert client.post(
        "/management/models/load", json={"name": "missing", "type": "llm"}
    ).status_code == 404


def test_management_unload_does_not_start_an_absent_model_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    models = tmp_path / "models"
    (models / "model-a").mkdir(parents=True)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(model_manager, "_metadata_path", tmp_path / "metadata.json")
    (tmp_path / "metadata.json").write_text('{"model-a":{"type":"llm"}}')
    supervisor = GenerationWorkerSupervisor()
    monkeypatch.setattr(supervisor, "_start_worker", lambda *args, **kwargs: pytest.fail("started"))
    app = create_app()
    app.state.generation_worker_supervisor = supervisor
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    assert client.post(
        "/management/models/unload", json={"name": "model-a", "type": "llm"}
    ).json() == {"unloaded": []}
    model = client.get("/management/state").json()["models"][0]
    assert (model["load_state"], model["request_state"]) == ("not_loaded", "idle")


def test_streaming_response_holds_worker_until_the_stream_is_closed(monkeypatch):
    from mlx_serve.routers.generation_proxy import WorkerStreamingResponse

    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    monkeypatch.setattr(settings, "generation_worker_idle_timeout_seconds", 1.0)
    supervisor = GenerationWorkerSupervisor()
    monkeypatch.setattr(supervisor, "_start_worker", fake_worker)

    class Stream:
        def __init__(self):
            self.reads = 0
            self.closed = False

        def read(self, size):
            worker = supervisor._workers["llm:model-a"]
            assert worker.active_leases == 1
            assert supervisor.reap_idle_workers(now=time.monotonic() + 60) == []
            self.reads += 1
            return b"data: done\n\n" if self.reads == 1 else b""

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setattr(
        "mlx_serve.routers.generation_proxy.open_worker_stream",
        lambda *args: WorkerStreamingResponse(200, {"Content-Type": "text/event-stream"}, stream),
    )
    app = create_app()
    app.state.generation_worker_supervisor = supervisor
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    response = client.post("/v1/completions", json={"model": "model-a", "stream": True})
    assert response.status_code == 200 and b"data: done" in response.content
    assert stream.closed
    assert supervisor._workers["llm:model-a"].active_leases == 0


@pytest.mark.asyncio
async def test_internal_model_scoped_worker_rejects_a_different_model(monkeypatch):
    from mlx_serve.core.runtime_topology import (
        GENERATION_WORKER_KIND_ENV,
        GENERATION_WORKER_MODEL_ENV,
    )
    from mlx_serve.routers.management import ModelOperation, _local_operation

    monkeypatch.setenv(GENERATION_WORKER_KIND_ENV, "llm")
    monkeypatch.setenv(GENERATION_WORKER_MODEL_ENV, "model-a")
    with pytest.raises(HTTPException) as error:
        await _local_operation("unload", ModelOperation(name="model-b", type="llm"))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_management_state_probes_unresponsive_workers_concurrently(monkeypatch):
    from mlx_serve.routers import management

    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_mode", "type")
    monkeypatch.setattr(management, "_local_snapshot", lambda: {})
    monkeypatch.setattr(management, "model_storage_available", lambda _: False)
    started = []
    all_started = asyncio.Event()

    async def unresponsive(url, _path):
        started.append(url)
        if len(started) == 5:
            all_started.set()
        await all_started.wait()
        raise HTTPException(status_code=503)

    monkeypatch.setattr(management, "_worker_request", unresponsive)
    supervisor = SimpleNamespace(snapshot=lambda: {
        kind: {"url": kind, "alive": True} for kind in ("llm", "vlm")
    })
    state = SimpleNamespace(
        retrieval_worker_urls={kind: kind for kind in ("embedding", "reranker", "decision")},
        generation_worker_supervisor=supervisor,
    )
    result = await asyncio.wait_for(
        management.management_state(SimpleNamespace(app=SimpleNamespace(state=state))), timeout=1,
    )
    assert len(started) == 5
    assert result["state"] == "degraded"
    assert all(not item["available"] for item in result["workers"].values())


@pytest.mark.asyncio
async def test_management_state_bounds_probes_and_keeps_reachable_workers(monkeypatch):
    from mlx_serve.routers import management

    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", False)
    monkeypatch.setattr(management, "_local_snapshot", lambda: {})
    monkeypatch.setattr(management, "model_storage_available", lambda _: False)
    monkeypatch.setattr(management, "_WORKER_STATE_DEADLINE_SECONDS", 0.05)
    cancelled = asyncio.Event()

    async def probe(url, _path):
        if url == "healthy":
            return {"cache": {}, "requests": {}}
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(management, "_worker_request", probe)
    state = SimpleNamespace(retrieval_worker_urls={"embedding": "healthy", "reranker": "slow"})
    result = await asyncio.wait_for(
        management.management_state(SimpleNamespace(app=SimpleNamespace(state=state))), timeout=1,
    )
    assert result["workers"]["embedding"]["available"]
    assert not result["workers"]["reranker"]["available"]
    assert not result["workers"]["decision"]["available"]
    assert cancelled.is_set()
