"""Tests for embeddings API."""

import base64

import numpy as np
import pytest


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
    assert "detail" in data
    assert data["detail"]["error"]["code"] == "model_not_found"


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
    assert "detail" in data
    assert data["detail"]["error"]["code"] == "invalid_input"


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

    # Test default dimensions (None)
    req = EmbeddingRequest(model="test", input="hello")
    assert req.dimensions is None

    # Test with dimensions
    req = EmbeddingRequest(model="test", input="hello", dimensions=256)
    assert req.dimensions == 256


def test_embeddings_request_dimensions_validation():
    """Test dimensions parameter validation."""
    from pydantic import ValidationError

    from mlx_serve.routers.embeddings import EmbeddingRequest

    # dimensions must be > 0
    with pytest.raises(ValidationError):
        EmbeddingRequest(model="test", input="hello", dimensions=0)

    with pytest.raises(ValidationError):
        EmbeddingRequest(model="test", input="hello", dimensions=-1)


# ============================================================================
# Helper function tests
# ============================================================================


def test_truncate_embedding():
    """Test truncate_embedding helper function."""
    from mlx_serve.routers.embeddings import truncate_embedding

    # Create a sample embedding (not normalized)
    original = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Truncate to 4 dimensions
    result = truncate_embedding(original, 4)

    # Check length
    assert len(result) == 4

    # Check L2 normalization (magnitude should be ~1)
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 0.001


def test_truncate_embedding_preserves_direction():
    """Test that truncation preserves relative proportions."""
    from mlx_serve.routers.embeddings import truncate_embedding

    original = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = truncate_embedding(original, 3)

    # Check that relative proportions are preserved after normalization
    # result[1] / result[0] should equal original[1] / original[0]
    assert abs(result[1] / result[0] - 2.0) < 0.001
    assert abs(result[2] / result[0] - 3.0) < 0.001


def test_encode_embedding_base64():
    """Test encode_embedding_base64 helper function."""
    from mlx_serve.routers.embeddings import encode_embedding_base64

    original = [0.1, 0.2, 0.3, 0.4]

    # Encode to base64
    encoded = encode_embedding_base64(original)

    # Should be a string
    assert isinstance(encoded, str)

    # Decode and verify
    decoded_bytes = base64.b64decode(encoded)
    decoded = np.frombuffer(decoded_bytes, dtype="<f4")

    assert len(decoded) == 4
    np.testing.assert_array_almost_equal(decoded, original, decimal=5)


def test_encode_embedding_base64_roundtrip():
    """Test base64 encoding roundtrip with various values."""
    from mlx_serve.routers.embeddings import encode_embedding_base64

    # Test with negative values and larger array
    original = [-0.5, 0.0, 0.5, 1.0, -1.0, 0.123456]

    encoded = encode_embedding_base64(original)
    decoded_bytes = base64.b64decode(encoded)
    decoded = np.frombuffer(decoded_bytes, dtype="<f4").tolist()

    # float32 has ~7 decimal digits precision
    for orig, dec in zip(original, decoded):
        assert abs(orig - dec) < 1e-6


# ============================================================================
# Integration tests (require mock or real model)
# ============================================================================


def test_embeddings_encoding_format_float_default(client, mock_embedding_model):
    """Test that default encoding format is float array."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
        },
    )

    if response.status_code == 200:
        data = response.json()
        embedding = data["data"][0]["embedding"]
        # Should be a list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)


def test_embeddings_encoding_format_base64(client, mock_embedding_model):
    """Test base64 encoding format."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "encoding_format": "base64",
        },
    )

    if response.status_code == 200:
        data = response.json()
        embedding = data["data"][0]["embedding"]

        # Should be a base64 string
        assert isinstance(embedding, str)

        # Should be valid base64
        decoded_bytes = base64.b64decode(embedding)
        decoded = np.frombuffer(decoded_bytes, dtype="<f4")
        assert len(decoded) > 0


def test_embeddings_dimensions_truncation(client, mock_embedding_model):
    """Test dimensions parameter truncation."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "dimensions": 128,
        },
    )

    if response.status_code == 200:
        data = response.json()
        embedding = data["data"][0]["embedding"]

        # Should be truncated to 128 dimensions
        assert len(embedding) == 128

        # Should be L2 normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001


def test_embeddings_dimensions_with_base64(client, mock_embedding_model):
    """Test dimensions with base64 encoding combined."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
            "dimensions": 64,
            "encoding_format": "base64",
        },
    )

    if response.status_code == 200:
        data = response.json()
        embedding_b64 = data["data"][0]["embedding"]

        # Decode base64
        decoded_bytes = base64.b64decode(embedding_b64)
        decoded = np.frombuffer(decoded_bytes, dtype="<f4")

        # Should be 64 dimensions
        assert len(decoded) == 64

        # Should be L2 normalized
        norm = np.linalg.norm(decoded)
        assert abs(norm - 1.0) < 0.001
