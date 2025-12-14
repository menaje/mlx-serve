"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from mlx_serve.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)
