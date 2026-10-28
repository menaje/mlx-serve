"""Tests for the isolated retrieval-worker HTTP proxy."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

import pytest
from fastapi.testclient import TestClient

from mlx_serve.config import settings
from mlx_serve.routers.retrieval_proxy import perform_worker_request
from mlx_serve.server import create_app


def test_worker_request_uses_bounded_timeout(monkeypatch):
    """Internal worker calls should never wait without a socket deadline."""
    observed: dict[str, object] = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout, **kwargs):
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(settings, "inference_queue_timeout_seconds", 12.5)
    monkeypatch.setattr(
        "mlx_serve.routers.retrieval_proxy.open_local_http",
        fake_urlopen,
    )

    response = perform_worker_request(
        "http://127.0.0.1:19080/v1/embeddings",
        "POST",
        b'{"model":"test"}',
        {"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert observed["timeout"] == 12.5


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
def test_decision_proxy_allows_execution_longer_than_queue_timeout(monkeypatch, path):
    """The gateway must not time out while a valid decision is still executing."""

    class SlowWorker(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(0.15)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b'{"ok":true}')))
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *args):
            pass

    class LocalWorkerServer(ThreadingHTTPServer):
        def server_bind(self):
            # Avoid a reverse DNS lookup for the loopback address.
            TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

    server = LocalWorkerServer(("127.0.0.1", 0), SlowWorker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)
        monkeypatch.setattr(settings, "inference_queue_timeout_seconds", 0.03)
        monkeypatch.setattr(settings, "decision_execution_timeout_seconds", 2.0)
        app = create_app()
        app.state.retrieval_worker_urls = {"decision": f"http://127.0.0.1:{server.server_port}"}
        client = TestClient(app)
        try:
            response = client.post(path, json={"model": "test"})
            assert response.status_code == 200
            assert response.json() == {"ok": True}
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_decision_proxy_preserves_unbounded_queue_wait(monkeypatch):
    observed = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout, **kwargs):
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", True)
    monkeypatch.setattr(settings, "inference_queue_timeout_seconds", None)
    monkeypatch.setattr("mlx_serve.routers.retrieval_proxy.open_local_http", fake_urlopen)
    app = create_app()
    app.state.retrieval_worker_urls = {"decision": "http://127.0.0.1:19080"}
    client = TestClient(app)
    try:
        response = client.post("/v1/decisions", json={"model": "test"})
        assert response.status_code == 200
        assert observed["timeout"] is None
    finally:
        client.close()
