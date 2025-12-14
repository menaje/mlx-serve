"""Tests for rerank API."""


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
    assert "error" in data
    assert data["error"]["code"] == "model_not_found"


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
    assert "error" in data
    assert data["error"]["code"] == "invalid_input"


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
