"""Tests for embeddings API."""


def test_embeddings_model_not_found(client):
    """Test embeddings endpoint with non-existent model."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "nonexistent-model",
            "input": "test text",
        },
    )
    assert response.status_code == 404

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "model_not_found"


def test_embeddings_empty_input(client):
    """Test embeddings endpoint with empty input."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": [],
        },
    )
    assert response.status_code == 400

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_input"


def test_embeddings_request_schema():
    """Test embeddings request schema validation."""
    from mlx_serve.routers.embeddings import EmbeddingRequest

    # Test with string input
    req = EmbeddingRequest(model="test", input="hello")
    assert req.input == "hello"

    # Test with list input
    req = EmbeddingRequest(model="test", input=["hello", "world"])
    assert req.input == ["hello", "world"]

    # Test default encoding format
    req = EmbeddingRequest(model="test", input="hello")
    assert req.encoding_format == "float"
