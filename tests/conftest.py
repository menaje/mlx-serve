"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_serve.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


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

    # Mock _generate_embeddings_batch to return random embeddings
    async def mock_generate_batch(model, tokenizer, texts):
        # Generate random embeddings with 512 dimensions
        return [np.random.randn(512).tolist() for _ in texts]

    # Patch model_manager.get_embedding_model and _generate_embeddings_batch
    with patch(
        "mlx_serve.core.model_manager.model_manager.get_embedding_model"
    ) as mock_get_model:
        mock_get_model.return_value = (mock_model, mock_tokenizer)

        with patch(
            "mlx_serve.routers.embeddings._generate_embeddings_batch",
            mock_generate_batch,
        ):
            yield mock_model, mock_tokenizer
