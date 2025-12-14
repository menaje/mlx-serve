"""Tests for tokenize API endpoint."""

import pytest


class TestTokenizeEndpoint:
    """Tests for /v1/tokenize endpoint."""

    def test_tokenize_model_not_found(self, client):
        """Test tokenize with non-existent model."""
        response = client.post(
            "/v1/tokenize",
            json={
                "model": "nonexistent-model",
                "input": "Hello world",
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]["error"]["message"]

    def test_tokenize_empty_input(self, client):
        """Test tokenize with empty input."""
        response = client.post(
            "/v1/tokenize",
            json={
                "model": "test-model",
                "input": [],
            },
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]["error"]["message"]

    def test_tokenize_request_validation(self, client):
        """Test tokenize request validation."""
        # Missing required field
        response = client.post(
            "/v1/tokenize",
            json={
                "input": "Hello world",
            },
        )
        assert response.status_code == 422  # Validation error


class TestTokenizeModels:
    """Tests for tokenize data models."""

    def test_tokenize_request_model(self):
        """Test TokenizeRequest model."""
        from mlx_serve.routers.tokenize import TokenizeRequest

        request = TokenizeRequest(
            model="test-model",
            input="Hello world",
            return_tokens=True,
        )
        assert request.model == "test-model"
        assert request.input == "Hello world"
        assert request.return_tokens is True

    def test_tokenize_request_list_input(self):
        """Test TokenizeRequest with list input."""
        from mlx_serve.routers.tokenize import TokenizeRequest

        request = TokenizeRequest(
            model="test-model",
            input=["Hello", "World"],
        )
        assert request.input == ["Hello", "World"]

    def test_token_data_model(self):
        """Test TokenData model."""
        from mlx_serve.routers.tokenize import TokenData

        data = TokenData(
            index=0,
            tokens=5,
            token_ids=[1, 2, 3, 4, 5],
        )
        assert data.index == 0
        assert data.tokens == 5
        assert data.token_ids == [1, 2, 3, 4, 5]

    def test_token_data_without_ids(self):
        """Test TokenData without token_ids."""
        from mlx_serve.routers.tokenize import TokenData

        data = TokenData(index=0, tokens=5)
        assert data.token_ids is None

    def test_tokenize_response_model(self):
        """Test TokenizeResponse model."""
        from mlx_serve.routers.tokenize import TokenData, TokenizeResponse

        response = TokenizeResponse(
            data=[TokenData(index=0, tokens=5)],
            model="test-model",
        )
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.model == "test-model"
