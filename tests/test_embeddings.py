"""Tests for embeddings API."""

import base64
from unittest.mock import MagicMock

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


def test_embeddings_clear_mlx_cache_after_request(client, mock_embedding_model, monkeypatch):
    """Embedding requests should clear MLX cache when the safeguard is enabled."""
    from mlx_serve.config import settings

    clear_mock = MagicMock()
    monkeypatch.setattr(settings, "retrieval_clear_mlx_cache_after_request", True)
    monkeypatch.setattr("mlx_serve.routers.embeddings.clear_mlx_cache", clear_mock)

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "test-model",
            "input": "test text",
        },
    )

    assert response.status_code == 200
    clear_mock.assert_called_once()


def test_unload_embedding_model_releases_batch_processor(monkeypatch):
    """Unloading an embedding model should stop its shared batch processor."""
    from mlx_serve.core.inference_control import build_inference_key
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.routers import embeddings as embeddings_router

    class FakeProcessor:
        def __init__(self):
            self.stopped = False
            self.closed = False

        def stop_nowait(self):
            self.stopped = True

        def close_nowait(self):
            self.closed = True

    processor = FakeProcessor()
    processor_key = build_inference_key("embedding", "embed-loaded")
    embeddings_router._batch_processors[processor_key] = processor
    model_manager._embedding_cache.set("embed-loaded", object())
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)

    unloaded = model_manager.unload_model("embed-loaded", "embedding")

    assert unloaded == [{"name": "embed-loaded", "type": "embedding"}]
    assert processor.closed is True
    assert processor_key not in embeddings_router._batch_processors


def test_expired_embedding_cache_access_releases_batch_processor(monkeypatch):
    """Expired embedding cache entries should run unload hooks before reload."""
    import time

    from mlx_serve.config import settings
    from mlx_serve.core.inference_control import build_inference_key
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.routers import embeddings as embeddings_router

    class FakeProcessor:
        def __init__(self):
            self.closed = False

        def close_nowait(self):
            self.closed = True

    processor = FakeProcessor()
    processor_key = build_inference_key("embedding", "embed-expired")
    embeddings_router._batch_processors[processor_key] = processor
    model_manager._embedding_cache.set("embed-expired", object())
    model_manager._embedding_cache._timestamps["embed-expired"] = (
        time.time() - settings.cache_ttl_seconds - 1
    )
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)

    expired = model_manager._cleanup_expired_cache(
        "embedding",
        model_manager._embedding_cache,
        "test expired embedding",
    )

    assert expired == ["embed-expired"]
    assert processor.closed is True
    assert processor_key not in embeddings_router._batch_processors


@pytest.mark.asyncio
async def test_embed_texts_stops_stale_processor(monkeypatch):
    """Replacing a model-backed embedding processor should stop the old task."""
    from mlx_serve.core.inference_control import build_inference_key
    from mlx_serve.routers import embeddings as embeddings_router

    old_model = object()
    old_tokenizer = object()
    new_model = object()
    new_tokenizer = object()

    class OldProcessor:
        model = old_model
        tokenizer = old_tokenizer

        def __init__(self):
            self.closed = False

        def close_nowait(self):
            self.closed = True

    class NewProcessor:
        def __init__(self, model, tokenizer, execution_lock=None):
            self.model = model
            self.tokenizer = tokenizer
            self.execution_lock = execution_lock

        async def embed(self, texts):
            return [[1.0] for _ in texts]

    old_processor = OldProcessor()
    processor_key = build_inference_key("embedding", "embed-loaded")
    embeddings_router._batch_processors[processor_key] = old_processor
    monkeypatch.setattr(embeddings_router, "EmbeddingBatchProcessor", NewProcessor)

    result = await embeddings_router._embed_texts(
        "embed-loaded",
        new_model,
        new_tokenizer,
        ["text"],
    )

    assert result == [[1.0]]
    assert old_processor.closed is True
    assert embeddings_router._batch_processors[processor_key].model is new_model
