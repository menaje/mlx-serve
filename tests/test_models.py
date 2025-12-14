"""Tests for models API."""


def test_list_models_openai(client):
    """Test OpenAI-compatible models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_list_models_ollama(client):
    """Test Ollama-compatible tags endpoint."""
    response = client.get("/api/tags")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_show_model_not_found(client):
    """Test show endpoint for non-existent model."""
    response = client.post("/api/show", json={"name": "nonexistent-model"})
    assert response.status_code == 404

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "model_not_found"


def test_delete_model_not_found(client):
    """Test delete endpoint for non-existent model."""
    response = client.request("DELETE", "/api/delete", json={"name": "nonexistent-model"})
    assert response.status_code == 404


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
