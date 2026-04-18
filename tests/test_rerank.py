"""Tests for rerank API."""

from unittest.mock import MagicMock


def test_rerank_model_not_found(client):
    """Test rerank endpoint with non-existent model."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "nonexistent-model",
            "query": "test query",
            "documents": ["doc1", "doc2"],
        },
    )
    assert response.status_code == 404

    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"]["code"] == "model_not_found"


def test_rerank_empty_documents(client):
    """Test rerank endpoint with empty documents list."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "test-model",
            "query": "test query",
            "documents": [],
        },
    )
    assert response.status_code == 400

    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"]["code"] == "invalid_input"


def test_rerank_request_schema():
    """Test rerank request schema validation."""
    from mlx_serve.routers.rerank import RerankRequest

    # Test with defaults
    req = RerankRequest(
        model="test",
        query="hello",
        documents=["doc1", "doc2"],
    )
    assert req.return_documents is True
    assert req.top_n is None

    # Test with custom values
    req = RerankRequest(
        model="test",
        query="hello",
        documents=["doc1", "doc2"],
        top_n=1,
        return_documents=False,
    )
    assert req.top_n == 1
    assert req.return_documents is False


def test_rerank_clear_mlx_cache_after_request(client, monkeypatch):
    """Rerank requests should clear MLX cache when the safeguard is enabled."""
    from mlx_serve.config import settings

    clear_mock = MagicMock()
    monkeypatch.setattr(settings, "retrieval_clear_mlx_cache_after_request", True)
    monkeypatch.setattr("mlx_serve.routers.rerank.clear_mlx_cache", clear_mock)
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.model_manager.get_reranker_model",
        lambda model: (object(), MagicMock()),
    )

    async def fake_compute_batch_scores(model, tokenizer, query, documents, instruction=None):
        return [0.9, 0.1], 8

    monkeypatch.setattr(
        "mlx_serve.routers.rerank._compute_batch_scores",
        fake_compute_batch_scores,
    )

    response = client.post(
        "/v1/rerank",
        json={
            "model": "test-model",
            "query": "test query",
            "documents": ["doc1", "doc2"],
        },
    )

    assert response.status_code == 200
    clear_mock.assert_called_once()
