"""Offline removal must preserve other registrations and shared originals."""

import json

import pytest
from fastapi.testclient import TestClient

from mlx_serve.core.offline_models import (
    offline_model_list,
    preview_model_removal,
    remove_offline_model,
)


@pytest.fixture
def storage(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("embed", "rerank"):
        target = models / name
        target.mkdir()
        (target / "weights").write_bytes(name.encode())
    originals = tmp_path / "decision-originals"
    originals.mkdir()
    (originals / "source").write_bytes(b"keep")
    registry = tmp_path / "metadata.json"
    data = {
        "embed": {"type": "embedding", "hf_repo": "example/embed", "size": 5},
        "rerank": {"type": "reranker", "hf_repo": "example/rerank", "size": 6},
    }
    registry.write_text(json.dumps(data))
    return models, registry, originals, data


def test_remove_registered_model_after_preview_keeps_other_model_and_original(storage):
    models, registry, originals, _ = storage
    assert [item["name"] for item in offline_model_list(models)["models"]] == ["embed", "rerank"]
    preview = preview_model_removal(models, "embed", "embedding")
    assert preview["size_bytes"] == 5
    assert remove_offline_model(models, preview) == {"removed": "embed", "size_bytes": 5}
    assert not (models / "embed").exists()
    assert (models / "rerank" / "weights").read_bytes() == b"rerank"
    assert (originals / "source").read_bytes() == b"keep"
    assert list(json.loads(registry.read_text())) == ["rerank"]


def test_removal_rechecks_file_registry_and_cross_model_references(storage):
    models, registry, _, data = storage
    preview = preview_model_removal(models, "embed", "embedding")
    (models / "embed" / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        remove_offline_model(models, preview)
    preview = preview_model_removal(models, "embed", "embedding")
    data["embed"]["hf_repo"] = "different/repo"
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        remove_offline_model(models, preview)
    data["rerank"]["original_path"] = str(models / "embed" / "weights")
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="references"):
        preview_model_removal(models, "embed", "embedding")
    assert (models / "embed" / "weights").read_bytes() == b"changed"


def test_removal_rejects_links_bad_registry_and_wrong_identity(storage, tmp_path):
    models, registry, _, _ = storage
    with pytest.raises(ValueError, match="not installed"):
        preview_model_removal(models, "embed", "reranker")
    with pytest.raises(ValueError, match="invalid"):
        preview_model_removal(models, "../outside", "embedding")
    (models / "embed" / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="safely checked"):
        preview_model_removal(models, "embed", "embedding")
    registry.write_text("bad")
    with pytest.raises(ValueError, match="registry"):
        offline_model_list(models)


def test_removal_rolls_back_if_registry_save_fails(storage, monkeypatch):
    import mlx_serve.core.offline_models as offline

    models, registry, _, _ = storage
    preview = preview_model_removal(models, "embed", "embedding")
    replace = offline.os.replace

    def reject_registry(source, destination):
        if destination == registry:
            raise OSError("simulated failure")
        return replace(source, destination)

    monkeypatch.setattr(offline.os, "replace", reject_registry)
    with pytest.raises(OSError, match="simulated"):
        remove_offline_model(models, preview)
    assert (models / "embed" / "weights").read_bytes() == b"embed"
    assert "embed" in json.loads(registry.read_text())
    assert not list(models.glob(".model-removal-*"))


def test_offline_model_operations_require_auth_and_stopped_server(tmp_path, monkeypatch):
    from mlx_serve.management_agent import ServerController, create_agent

    monkeypatch.setattr("mlx_serve.management_agent.initialize_app_config", lambda: None)

    async def running(self):
        return {"state": "connected", "ownership": "app"}

    monkeypatch.setattr(ServerController, "status", running)
    app = create_agent("private-secret", data_dir=tmp_path)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 12345)) as client:
        assert client.get("/stored-models").status_code == 403
        response = client.get("/stored-models", headers={"Authorization": "Bearer private-secret"})
        assert response.status_code == 409
        assert "Stop the server" in response.json()["detail"]
