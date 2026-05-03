"""Tests for gateway/worker runtime topology."""

import pytest
from fastapi.testclient import TestClient

from mlx_serve.config import settings
from mlx_serve.core.runtime_topology import (
    GENERATION_WORKER_KIND_ENV,
    GENERATION_WORKER_MODEL_ENV,
    RETRIEVAL_WORKER_KIND_ENV,
    SERVER_ROLE_ENV,
)
from mlx_serve.routers.generation_proxy import WorkerStreamingResponse
from mlx_serve.routers.retrieval_proxy import WorkerProxyResponse
from mlx_serve.server import create_app


def _post_endpoint_module(app, path: str) -> str:
    for route in app.routes:
        methods = getattr(route, "methods", set())
        if route.path == path and "POST" in methods:
            return route.endpoint.__module__
    raise AssertionError(f"Route not found: {path}")


def test_gateway_uses_proxy_routes_when_retrieval_isolation_enabled(monkeypatch):
    """Gateway mode should mount proxy routes for retrieval endpoints."""
    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)

    app = create_app()

    assert _post_endpoint_module(app, "/v1/embeddings") == "mlx_serve.routers.retrieval_proxy"
    assert _post_endpoint_module(app, "/v1/rerank") == "mlx_serve.routers.retrieval_proxy"


def test_gateway_uses_proxy_routes_when_generation_isolation_enabled(monkeypatch):
    """Gateway mode should mount proxy routes for generation endpoints."""
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)

    app = create_app()

    assert _post_endpoint_module(app, "/v1/chat/completions") == (
        "mlx_serve.routers.generation_proxy"
    )
    assert _post_endpoint_module(app, "/v1/completions") == "mlx_serve.routers.generation_proxy"


def test_embedding_worker_only_exposes_embedding_routes(monkeypatch):
    """Embedding workers should only mount the embedding endpoint."""
    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(RETRIEVAL_WORKER_KIND_ENV, "embedding")

    app = create_app()
    paths = {route.path for route in app.routes}

    assert "/v1/embeddings" in paths
    assert "/v1/tokenize" in paths
    assert "/v1/rerank" not in paths
    assert "/v1/chat/completions" not in paths


def test_llm_worker_only_exposes_generation_routes(monkeypatch):
    """LLM workers should mount chat/completion endpoints without retrieval routes."""
    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(GENERATION_WORKER_KIND_ENV, "llm")

    app = create_app()
    paths = {route.path for route in app.routes}

    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
    assert "/v1/embeddings" not in paths
    assert "/v1/rerank" not in paths


def test_model_scoped_generation_worker_rejects_unassigned_model(monkeypatch):
    """A model-scoped generation worker must not serve a different model."""
    from fastapi import HTTPException

    from mlx_serve.routers.chat import _ensure_generation_worker_kind

    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(GENERATION_WORKER_KIND_ENV, "llm")
    monkeypatch.setenv(GENERATION_WORKER_MODEL_ENV, "llama-3.2-1b")

    with pytest.raises(HTTPException) as exc_info:
        _ensure_generation_worker_kind("llm", "qwen2.5-3b")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["code"] == "model_not_found"


def test_worker_health_includes_mlx_memory_snapshot(monkeypatch):
    """Worker health should expose MLX runtime counters for diagnostics."""
    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(RETRIEVAL_WORKER_KIND_ENV, "embedding")
    monkeypatch.setattr(
        "mlx_serve.server.get_mlx_memory_snapshot",
        lambda: {
            "available": True,
            "active_bytes": 11,
            "cache_bytes": 22,
            "peak_bytes": 33,
        },
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["mlx_memory"] == {
        "available": True,
        "active_bytes": 11,
        "cache_bytes": 22,
        "peak_bytes": 33,
    }


def test_generation_worker_health_reports_kind(monkeypatch):
    """Generation worker health should include the worker kind."""
    monkeypatch.setenv(SERVER_ROLE_ENV, "worker")
    monkeypatch.setenv(GENERATION_WORKER_KIND_ENV, "vlm")
    monkeypatch.setenv(GENERATION_WORKER_MODEL_ENV, "Qwen2-VL-2B-Instruct-4bit")

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["generation_worker_kind"] == "vlm"
    assert response.json()["generation_worker_model"] == "Qwen2-VL-2B-Instruct-4bit"


def test_gateway_proxies_embedding_requests(monkeypatch):
    """Gateway requests should be forwarded to the isolated embedding worker."""
    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)

    calls: dict[str, object] = {}

    class FakeSupervisor:
        def start(self):
            return {
                "embedding": "http://127.0.0.1:19080",
                "reranker": "http://127.0.0.1:19081",
            }

        def stop(self):
            return None

        def snapshot(self):
            return {
                "embedding": {"url": "http://127.0.0.1:19080", "pid": 101, "alive": True},
                "reranker": {"url": "http://127.0.0.1:19081", "pid": 102, "alive": True},
            }

    def fake_worker_request(url: str, method: str, body: bytes, headers: dict[str, str]):
        calls["url"] = url
        calls["method"] = method
        calls["body"] = body
        calls["headers"] = headers
        return WorkerProxyResponse(
            status_code=200,
            body=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("mlx_serve.server.RetrievalWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.retrieval_proxy.perform_worker_request",
        fake_worker_request,
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "Qwen3-Embedding-0.6B", "input": "hello"},
        )
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls["url"] == "http://127.0.0.1:19080/v1/embeddings"
    assert calls["method"] == "POST"
    assert b'"model":"Qwen3-Embedding-0.6B"' in calls["body"]
    assert calls["headers"] == {"Content-Type": "application/json"}
    assert health.status_code == 200
    assert health.json()["retrieval_workers"]["embedding"]["alive"] is True


def test_gateway_proxies_chat_requests_to_generation_worker(monkeypatch):
    """Gateway chat requests should route to the LLM or VLM worker."""
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)

    calls: list[str] = []

    class FakeSupervisor:
        def start(self):
            return {
                "llm": "http://127.0.0.1:19180",
                "vlm": "http://127.0.0.1:19181",
            }

        def stop(self):
            return None

        def snapshot(self):
            return {
                "llm": {"url": "http://127.0.0.1:19180", "pid": 201, "alive": True},
                "vlm": {"url": "http://127.0.0.1:19181", "pid": 202, "alive": True},
            }

    def fake_worker_request(url: str, method: str, body: bytes, headers: dict[str, str]):
        calls.append(url)
        return WorkerProxyResponse(
            status_code=200,
            body=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("mlx_serve.server.GenerationWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.generation_proxy.perform_worker_request",
        fake_worker_request,
    )

    app = create_app()
    with TestClient(app) as client:
        text_response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "hi"}]},
        )
        image_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.2-1b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ],
            },
        )
        health = client.get("/health")

    assert text_response.status_code == 200
    assert image_response.status_code == 200
    assert calls == [
        "http://127.0.0.1:19180/v1/chat/completions",
        "http://127.0.0.1:19181/v1/chat/completions",
    ]
    assert health.json()["generation_workers"]["llm"]["alive"] is True


def test_gateway_starts_model_scoped_generation_worker_on_demand(monkeypatch):
    """Model-scoped generation mode should create workers per requested model."""
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "generation_worker_mode", "model")

    ensured: list[tuple[str, str]] = []
    calls: list[str] = []

    class FakeSupervisor:
        def start(self):
            return {}

        def ensure_worker(self, kind: str, model_name: str):
            ensured.append((kind, model_name))
            return "http://127.0.0.1:19280"

        def stop(self):
            return None

        def snapshot(self):
            return {
                "llm:Llama-3.2-1B-Instruct-4bit": {
                    "url": "http://127.0.0.1:19280",
                    "pid": 301,
                    "alive": True,
                    "model": "Llama-3.2-1B-Instruct-4bit",
                }
            }

    def fake_worker_request(url: str, method: str, body: bytes, headers: dict[str, str]):
        calls.append(url)
        return WorkerProxyResponse(
            status_code=200,
            body=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("mlx_serve.server.GenerationWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.generation_proxy.perform_worker_request",
        fake_worker_request,
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "hi"}]},
        )
        health = client.get("/health")

    assert response.status_code == 200
    assert ensured == [("llm", "llama-3.2-1b")]
    assert calls == ["http://127.0.0.1:19280/v1/chat/completions"]
    assert health.json()["generation_workers"]["llm:Llama-3.2-1B-Instruct-4bit"]["model"] == (
        "Llama-3.2-1B-Instruct-4bit"
    )


def test_model_scoped_generation_worker_reaps_idle_workers(monkeypatch):
    """Model-scoped generation workers should expire after the idle timeout."""
    from mlx_serve.core.generation_workers import (
        GenerationWorkerProcess,
        GenerationWorkerSupervisor,
    )

    class FakeProcess:
        pid = 1234

        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    monkeypatch.setattr(settings, "generation_worker_mode", "model")
    monkeypatch.setattr(settings, "generation_worker_idle_timeout_seconds", 5.0)

    supervisor = GenerationWorkerSupervisor()
    process = FakeProcess()
    worker = GenerationWorkerProcess(
        kind="llm",
        host="127.0.0.1",
        port=19000,
        process=process,
        model_name="model-a",
        last_used_at=10.0,
    )
    supervisor._workers["llm:model-a"] = worker

    reaped = supervisor.reap_idle_workers(now=16.0)

    assert reaped == ["llm:model-a"]
    assert process.terminated is True
    assert supervisor.snapshot() == {}


def test_gateway_streams_generation_worker_response(monkeypatch):
    """Streaming generation proxy should not use the buffered worker request path."""
    monkeypatch.setattr(settings, "generation_worker_isolation_enabled", True)

    class FakeSupervisor:
        def start(self):
            return {"llm": "http://127.0.0.1:19180", "vlm": "http://127.0.0.1:19181"}

        def stop(self):
            return None

        def snapshot(self):
            return {}

    class FakeResponse:
        def __init__(self):
            self._chunks = [b"data: one\n\n", b"data: [DONE]\n\n", b""]
            self.closed = False

        def read(self, _size):
            return self._chunks.pop(0)

        def close(self):
            self.closed = True

    def fail_buffered_request(*_args, **_kwargs):
        raise AssertionError("buffered request should not be used for stream responses")

    def fake_open_stream(*_args, **_kwargs):
        return WorkerStreamingResponse(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            response=FakeResponse(),
        )

    monkeypatch.setattr("mlx_serve.server.GenerationWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.generation_proxy.perform_worker_request",
        fail_buffered_request,
    )
    monkeypatch.setattr("mlx_serve.routers.generation_proxy.open_worker_stream", fake_open_stream)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.2-1b",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert "data: one" in response.text


def test_gateway_proxies_tokenize_requests_to_matching_worker(monkeypatch):
    """Gateway tokenize requests should route to the worker that owns the model type."""
    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)

    calls: dict[str, object] = {}

    class FakeSupervisor:
        def start(self):
            return {
                "embedding": "http://127.0.0.1:19080",
                "reranker": "http://127.0.0.1:19081",
            }

        def stop(self):
            return None

        def snapshot(self):
            return {}

    def fake_worker_request(url: str, method: str, body: bytes, headers: dict[str, str]):
        calls["url"] = url
        calls["method"] = method
        calls["body"] = body
        calls["headers"] = headers
        return WorkerProxyResponse(
            status_code=200,
            body=(
                b'{"object":"list","data":[{"index":0,"tokens":2,"token_ids":[1,2]}],'
                b'"model":"Qwen3-Reranker-0.6B"}'
            ),
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("mlx_serve.server.RetrievalWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.retrieval_proxy.perform_worker_request",
        fake_worker_request,
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/tokenize",
            json={"model": "Qwen3-Reranker-0.6B", "input": "hello", "return_tokens": True},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "Qwen3-Reranker-0.6B"
    assert calls["url"] == "http://127.0.0.1:19081/v1/tokenize"
    assert calls["method"] == "POST"
    assert b'"model":"Qwen3-Reranker-0.6B"' in calls["body"]
    assert calls["headers"] == {"Content-Type": "application/json"}


def test_gateway_tokenize_probes_both_workers_when_model_type_unknown(monkeypatch):
    """Gateway tokenize should probe both workers when model type cannot be inferred."""
    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)

    calls: list[str] = []

    class FakeSupervisor:
        def start(self):
            return {
                "embedding": "http://127.0.0.1:19080",
                "reranker": "http://127.0.0.1:19081",
            }

        def stop(self):
            return None

        def snapshot(self):
            return {}

    def fake_worker_request(url: str, method: str, body: bytes, headers: dict[str, str]):
        calls.append(url)
        if "19080" in url:
            return WorkerProxyResponse(
                status_code=404,
                body=b'{"error":{"message":"Model not found"}}',
                headers={"Content-Type": "application/json"},
            )
        return WorkerProxyResponse(
            status_code=200,
            body=b'{"object":"list","data":[{"index":0,"tokens":1}],"model":"manual-rerank"}',
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("mlx_serve.server.RetrievalWorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(
        "mlx_serve.routers.tokenize.resolve_retrieval_model_type",
        lambda model: None,
    )
    monkeypatch.setattr(
        "mlx_serve.routers.retrieval_proxy.perform_worker_request",
        fake_worker_request,
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/tokenize",
            json={"model": "manual-rerank", "input": "hello"},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "manual-rerank"
    assert calls == [
        "http://127.0.0.1:19080/v1/tokenize",
        "http://127.0.0.1:19081/v1/tokenize",
    ]
