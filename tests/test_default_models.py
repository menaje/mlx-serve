"""Default selection is per type, durable, and never selects missing assets."""

import json

import pytest
from pydantic import ValidationError

from mlx_serve.config import settings
from mlx_serve.core.default_models import default_model
from mlx_serve.core.model_manager import model_manager
from mlx_serve.routers.decision import DecisionRequest
from mlx_serve.routers.embeddings import EmbeddingRequest
from mlx_serve.routers.rerank import RerankRequest


@pytest.fixture
def installed(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    directory.mkdir()
    registry = {}
    for name, kind in (("decision-one", "decision"), ("embed-one", "embedding"),
                       ("rerank-one", "reranker")):
        (directory / name).mkdir()
        registry[name] = {
            "type": kind, "hf_repo": f"example/{name}", "revision": "revision-a",
            "modified_at": "2026-01-01", "size": 0,
        }
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(registry))
    monkeypatch.setattr(settings, "models_dir", directory)
    monkeypatch.setattr(model_manager, "_metadata_path", metadata)
    monkeypatch.setattr(model_manager, "_metadata", {})
    return directory, registry


def test_defaults_are_used_only_when_the_request_omits_the_model(installed, client, monkeypatch):
    directory, _ = installed
    for name, kind in (("decision-one", "decision"), ("embed-one", "embedding"),
                       ("rerank-one", "reranker")):
        response = client.post("/management/models/set-default", json={"name": name, "type": kind})
        assert response.status_code == 200
    models = client.get("/management/state").json()["models"]
    selected = {model["name"]: model["is_default"] for model in models}
    assert selected == {"decision-one": True, "embed-one": True, "rerank-one": True}
    assert (directory.parent / "default-models.json").stat().st_mode & 0o077 == 0

    decision = {"state": "input", "questions": {"q": {"type": "bool", "instructions": "question"}}}
    assert DecisionRequest.model_validate(decision).model == "decision-one"
    assert DecisionRequest.model_validate({**decision, "model": "explicit"}).model == "explicit"
    assert EmbeddingRequest.model_validate({"input": "input"}).model == "embed-one"
    assert RerankRequest.model_validate({"query": "q", "documents": ["d"]}).model == "rerank-one"
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({**decision, "model": None})

    monkeypatch.setattr(
        "mlx_serve.routers.decision._run_decision",
        lambda request, typed: {"answers": {}, "usage": {}},
    )
    result = client.post("/v1/decisions", json=decision)
    assert result.status_code == 200 and result.json()["model"] == "decision-one"
    result = client.post("/v1/decisions", json={**decision, "model": "explicit"})
    assert result.status_code == 200 and result.json()["model"] == "explicit"

    assert client.post(
        "/management/models/clear-default", json={"name": "other", "type": "decision"},
    ).status_code == 409
    assert client.post(
        "/management/models/clear-default", json={"name": "decision-one", "type": "decision"},
    ).status_code == 200
    assert default_model("decision") is None
    assert default_model("embedding") == "embed-one"
    assert client.post("/v1/decisions", json=decision).status_code == 422


def test_missing_or_changed_model_cannot_silently_remain_default(installed, client):
    directory, registry = installed
    assert client.post(
        "/management/models/set-default", json={"name": "decision-one", "type": "decision"},
    ).status_code == 200
    (directory / "decision-one").rmdir()
    assert default_model("decision") is None
    (directory / "decision-one").mkdir()
    registry["decision-one"]["revision"] = "revision-b"
    (directory.parent / "metadata.json").write_text(json.dumps(registry))
    assert default_model("decision") is None
    assert client.post(
        "/management/models/set-default", json={"name": "missing", "type": "decision"},
    ).status_code == 422


def test_corrupt_default_settings_are_not_overwritten(installed, client):
    directory, _ = installed
    path = directory.parent / "default-models.json"
    path.write_text("broken json")
    response = client.post(
        "/management/models/set-default", json={"name": "decision-one", "type": "decision"},
    )
    assert response.status_code == 422
    assert path.read_text() == "broken json"
    assert default_model("decision") is None


@pytest.mark.asyncio
async def test_deleting_a_decision_model_clears_its_default(installed, client):
    from mlx_serve.routers.decision import delete_decision_model

    directory, _ = installed
    assert client.post(
        "/management/models/set-default", json={"name": "decision-one", "type": "decision"},
    ).status_code == 200
    assert await delete_decision_model("decision-one")
    assert "decision" not in json.loads((directory.parent / "default-models.json").read_text())
    assert default_model("decision") is None
