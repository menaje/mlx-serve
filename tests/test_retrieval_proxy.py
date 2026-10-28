"""Tests for the isolated retrieval-worker HTTP proxy."""

from mlx_serve.config import settings
from mlx_serve.routers.retrieval_proxy import perform_worker_request


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

    def fake_urlopen(request, timeout):
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(settings, "inference_queue_timeout_seconds", 12.5)
    monkeypatch.setattr(
        "mlx_serve.routers.retrieval_proxy.urllib.request.urlopen",
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
