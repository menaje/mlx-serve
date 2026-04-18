"""Tests for gateway/worker runtime topology."""

from fastapi.testclient import TestClient

from mlx_serve.config import settings
from mlx_serve.core.runtime_topology import RETRIEVAL_WORKER_KIND_ENV, SERVER_ROLE_ENV
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
    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", fake_worker_request)

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
    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", fake_worker_request)

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
    monkeypatch.setattr("mlx_serve.routers.tokenize.resolve_retrieval_model_type", lambda model: None)
    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.perform_worker_request", fake_worker_request)

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
