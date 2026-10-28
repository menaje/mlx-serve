"""Only explicitly selected, unchanged and unregistered cache revisions are removed."""

import json

import pytest
from fastapi.testclient import TestClient

from mlx_serve.core.hf_cache_cleanup import (
    cleanup_stale_hf_cache_revision,
    stale_hf_cache_revisions,
)


@pytest.fixture
def cache_fixture(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    registry = tmp_path / "metadata.json"
    registry.write_text("{}")
    root = tmp_path / "hub"
    repo = root / "models--example--source"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "shared").write_bytes(b"shared")
    (blobs / "old").write_bytes(b"old file")
    (blobs / "current").write_bytes(b"current file")
    old, current = "a" * 40, "b" * 40
    for revision, blob in ((old, "old"), (current, "current")):
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "shared").symlink_to("../../blobs/shared")
        (snapshot / "unique").symlink_to(f"../../blobs/{blob}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(current)
    return models, registry, root, repo, old, current


def test_cleanup_deletes_only_detached_revision_and_unshared_blob(cache_fixture):
    models, _, root, repo, old, current = cache_fixture
    candidates = stale_hf_cache_revisions(models, root)["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["repo"] == "example/source"
    assert candidates[0]["revision"] == old
    assert candidates[0]["size_bytes"] == len(b"old file")

    result = cleanup_stale_hf_cache_revision(models, candidates[0], root)
    assert result == {"removed": 1, "estimated_bytes": len(b"old file")}
    assert not (repo / "snapshots" / old).exists()
    assert not (repo / "blobs" / "old").exists()
    assert (repo / "blobs" / "shared").read_bytes() == b"shared"
    assert (repo / "snapshots" / current / "shared").read_bytes() == b"shared"
    assert (repo / "refs" / "main").read_text() == current


def test_registered_sources_and_original_paths_protect_all_revisions(cache_fixture):
    models, registry, root, repo, old, _ = cache_fixture
    preview = stale_hf_cache_revisions(models, root)["candidates"][0]
    registry.write_text(json.dumps({"used": {"hf_repo": "EXAMPLE/SOURCE"}}))
    assert stale_hf_cache_revisions(models, root)["candidates"] == []
    with pytest.raises(ValueError, match="changed"):
        cleanup_stale_hf_cache_revision(models, preview, root)
    assert (repo / "snapshots" / old).exists()
    registry.write_text(json.dumps({"used": {"original_path": str(repo / "snapshots" / old)}}))
    assert stale_hf_cache_revisions(models, root)["candidates"] == []


def test_changed_cache_or_new_ref_invalidates_preview(cache_fixture):
    models, _, root, repo, old, _ = cache_fixture
    preview = stale_hf_cache_revisions(models, root)["candidates"][0]
    (repo / "blobs" / "old").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cleanup_stale_hf_cache_revision(models, preview, root)
    preview = stale_hf_cache_revisions(models, root)["candidates"][0]
    (repo / "refs" / "tag").write_text(old)
    with pytest.raises(ValueError, match="changed"):
        cleanup_stale_hf_cache_revision(models, preview, root)
    assert (repo / "snapshots" / old).exists()
    assert (repo / "blobs" / "old").read_bytes() == b"changed"


def test_symlinked_parent_is_canonicalized_but_linked_cache_root_is_rejected(cache_fixture):
    models, _, root, _, _, _ = cache_fixture
    alias = root.parent / "parent-alias"
    alias.symlink_to(root.parent, target_is_directory=True)
    assert len(stale_hf_cache_revisions(models, alias / root.name)["candidates"]) == 1
    root_link = root.parent / "cache-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="cache path"):
        stale_hf_cache_revisions(models, root_link)


def test_corrupt_registry_and_cache_fail_closed(cache_fixture):
    models, registry, root, repo, old, _ = cache_fixture
    preview = stale_hf_cache_revisions(models, root)["candidates"][0]
    registry.write_text("broken")
    with pytest.raises(ValueError, match="registry"):
        cleanup_stale_hf_cache_revision(models, preview, root)
    registry.write_text("{}")
    (root / "models--broken--cache").mkdir()
    with pytest.raises(ValueError, match="unreadable"):
        cleanup_stale_hf_cache_revision(models, preview, root)
    assert (repo / "snapshots" / old).exists()


def test_agent_cleanup_is_authorized_and_requires_stopped_server(tmp_path, monkeypatch):
    from mlx_serve.management_agent import ServerController, create_agent

    monkeypatch.setattr("mlx_serve.management_agent.initialize_app_config", lambda: None)

    async def running(self):
        return {"state": "connected", "ownership": "app"}

    monkeypatch.setattr(ServerController, "status", running)
    app = create_agent("private-secret", data_dir=tmp_path)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 12345)) as client:
        assert client.get("/hf-cache/cleanup").status_code == 403
        response = client.get(
            "/hf-cache/cleanup", headers={"Authorization": "Bearer private-secret"}
        )
        assert response.status_code == 409
        assert "Stop the server" in response.json()["detail"]
