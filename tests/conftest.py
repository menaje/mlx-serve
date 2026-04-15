"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_serve.config import settings
from mlx_serve.core.inference_control import inference_controller
from mlx_serve.core.runtime_topology import RETRIEVAL_WORKER_KIND_ENV, SERVER_ROLE_ENV
from mlx_serve.core.system_guard import memory_monitor
from mlx_serve.routers import embeddings as embeddings_router
from mlx_serve.server import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch):
    """Reset singleton runtime state between tests."""
    monkeypatch.setattr(settings, "memory_guard_enabled", False)
    monkeypatch.setattr(settings, "retrieval_worker_isolation_enabled", False)
    monkeypatch.delenv(SERVER_ROLE_ENV, raising=False)
    monkeypatch.delenv(RETRIEVAL_WORKER_KIND_ENV, raising=False)
    inference_controller.reset()
    embeddings_router._batch_processors.clear()
    memory_monitor.reset()
    yield
    inference_controller.reset()
    embeddings_router._batch_processors.clear()
    memory_monitor.reset()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing without actual model loading.

    Returns embeddings with 512 dimensions for testing.
    """
    # Create mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer.encode to return token list
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

    # Mock batched embedding helper to return random embeddings
    async def mock_embed_texts(model_name, model, tokenizer, texts):
        # Generate random embeddings with 512 dimensions
        return [np.random.randn(512).tolist() for _ in texts]

    # Patch model_manager.get_embedding_model and the embedding helper
    with patch(
        "mlx_serve.core.model_manager.model_manager.get_embedding_model"
    ) as mock_get_model:
        mock_get_model.return_value = (mock_model, mock_tokenizer)

        with patch(
            "mlx_serve.routers.embeddings._embed_texts",
            mock_embed_texts,
        ):
            yield mock_model, mock_tokenizer
