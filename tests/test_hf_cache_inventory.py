"""Shared cache inspection is read-only and works without a running engine."""

from fastapi.testclient import TestClient

from mlx_serve.core.hf_cache_inventory import inspect_hf_cache


def test_inventory_counts_shared_revisions_without_removing_any_files(tmp_path):
    cache = tmp_path / "hub"
    repo = cache / "models--example--model"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "filehash"
    blob.write_bytes(b"shared file")
    revisions = ["a" * 40, "b" * 40]
    for revision in revisions:
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").symlink_to("../../blobs/filehash")
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text(revisions[0])
    (cache / "models--broken--repo").mkdir()

    inventory = inspect_hf_cache(cache)

    assert inventory["size_bytes"] == len(b"shared file")
    assert inventory["repository_count"] == 1
    assert inventory["warning_count"] == 1
    assert inventory["repositories"] == [{
        "id": "example/model", "type": "model", "size_bytes": len(b"shared file"),
        "revision_count": 2,
    }]
    assert blob.read_bytes() == b"shared file"
    assert all((repo / "snapshots" / revision / "config.json").is_symlink()
               for revision in revisions)
    assert (refs / "main").read_text() == revisions[0]


def test_missing_cache_is_reported_as_empty(tmp_path):
    root = tmp_path / "missing"
    result = inspect_hf_cache(root)
    assert result["path"] == str(root)
    assert result["repository_count"] == 0 and result["size_bytes"] == 0
    assert not root.exists()


def test_agent_cache_inventory_requires_local_authorization(tmp_path, monkeypatch):
    from mlx_serve.management_agent import create_agent

    monkeypatch.setattr("mlx_serve.management_agent.initialize_app_config", lambda: None)
    expected = {"repository_count": 0, "repositories": []}
    monkeypatch.setattr("mlx_serve.core.hf_cache_inventory.inspect_hf_cache", lambda: expected)
    app = create_agent("private-secret", data_dir=tmp_path)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 12345)) as client:
        assert client.get("/hf-cache").status_code == 403
        response = client.get("/hf-cache", headers={"Authorization": "Bearer private-secret"})
        assert response.status_code == 200 and response.json() == expected
