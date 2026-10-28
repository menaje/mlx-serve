"""Cleanup only unreferenced, unchanged source directories inside model storage."""

import fcntl
import json

import pytest

from mlx_serve.config import settings


@pytest.fixture
def originals(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    directory.mkdir()
    root = tmp_path / "decision-originals"
    root.mkdir()
    names = {kind: f"{kind}-{'a' * 40}" for kind in ("old", "used", "linked", "nested")}
    for name in names.values():
        (root / name).mkdir()
        (root / name / "weights").write_bytes(b"12345")
    (root / names["nested"] / "link").symlink_to(tmp_path)
    (root / names["linked"]).rename(root / "outside-name")
    (root / names["linked"]).symlink_to(root / "outside-name")
    (root / "user-files").mkdir()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "one": {"original_path": str(root / names["used"])},
        "two": {"original_path": str(root / names["used"] / "weights")},
    }))
    monkeypatch.setattr(settings, "models_dir", directory)
    return directory, root, names, metadata


def test_cleanup_rechecks_preview_and_preserves_referenced_and_linked_files(originals, client):
    _, root, names, _ = originals
    response = client.get("/management/originals")
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert [entry["name"] for entry in candidates] == [names["old"]]
    assert candidates[0]["size_bytes"] == 5

    duplicate = client.post("/management/originals/cleanup", json={"entries": candidates * 2})
    assert duplicate.status_code == 409
    assert (root / names["old"] / "weights").read_bytes() == b"12345"
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 200 and result.json()["removed"] == 1
    assert not (root / names["old"]).exists()
    assert (root / names["used"] / "weights").read_bytes() == b"12345"
    assert (root / "outside-name" / "weights").read_bytes() == b"12345"
    assert (root / names["nested"] / "link").is_symlink()
    assert (root / "user-files").is_dir()


def test_cleanup_rejects_changed_files_and_new_references(originals, client):
    _, root, names, metadata = originals
    candidates = client.get("/management/originals").json()["candidates"]
    (root / names["old"] / "weights").write_bytes(b"updated")
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 409
    assert (root / names["old"] / "weights").read_bytes() == b"updated"

    candidates = client.get("/management/originals").json()["candidates"]
    data = json.loads(metadata.read_text())
    data["new"] = {"original_path": str(root / names["old"])}
    metadata.write_text(json.dumps(data))
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 409
    assert (root / names["old"] / "weights").read_bytes() == b"updated"


def test_cleanup_fails_closed_if_registry_is_corrupt_or_preparation_is_active(originals, client):
    directory, root, names, metadata = originals
    metadata.write_text("broken")
    assert client.get("/management/originals").status_code == 409
    assert (root / names["old"]).is_dir()
    metadata.write_text("{}")
    lock_dir = directory / ".decision-locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / "d1-preparation.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        assert client.get("/management/originals").status_code == 409
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    assert (root / names["old"]).is_dir()


def test_cleanup_does_not_follow_a_symlinked_original_root(tmp_path, monkeypatch, client):
    directory = tmp_path / "models"
    directory.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve")
    (tmp_path / "decision-originals").symlink_to(outside)
    (tmp_path / "metadata.json").write_text("{}")
    monkeypatch.setattr(settings, "models_dir", directory)
    assert client.get("/management/originals").status_code == 409
    assert (outside / "sentinel").read_text() == "preserve"


def test_cleanup_rolls_back_moves_if_preparation_for_deletion_fails(originals, client, monkeypatch):
    import mlx_serve.core.managed_originals as cleanup

    _, root, names, _ = originals
    second = root / f"extra-{'c' * 40}"
    second.mkdir()
    (second / "weights").write_bytes(b"second")
    candidates = client.get("/management/originals").json()["candidates"]
    assert len(candidates) == 2
    replace = cleanup.os.replace
    calls = 0

    def fail_second_move(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated rename failure")
        return replace(source, destination)

    monkeypatch.setattr(cleanup.os, "replace", fail_second_move)
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 409
    assert (root / names["old"] / "weights").read_bytes() == b"12345"
    assert (second / "weights").read_bytes() == b"second"
    assert not list(root.glob(".cleanup-*"))


def test_cleanup_reports_location_when_staged_deletion_fails(originals, client, monkeypatch):
    import mlx_serve.core.managed_originals as cleanup

    _, root, names, _ = originals
    candidates = client.get("/management/originals").json()["candidates"]

    def fail_removal(path):
        raise OSError("simulated deletion failure")

    monkeypatch.setattr(cleanup.shutil, "rmtree", fail_removal)
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 409
    assert "Cleanup was incomplete; inspect " in result.json()["detail"]
    staging = list(root.glob(".cleanup-*"))
    assert len(staging) == 1
    assert (staging[0] / names["old"] / "weights").read_bytes() == b"12345"


def test_cleanup_rechecks_files_after_they_are_moved(originals, client, monkeypatch):
    import mlx_serve.core.managed_originals as cleanup

    _, root, names, _ = originals
    candidates = client.get("/management/originals").json()["candidates"]
    replace = cleanup.os.replace
    changed = False

    def change_before_move(source, destination):
        nonlocal changed
        if source == root / names["old"] and not changed:
            (source / "weights").write_bytes(b"changed during cleanup")
            changed = True
        return replace(source, destination)

    monkeypatch.setattr(cleanup.os, "replace", change_before_move)
    result = client.post("/management/originals/cleanup", json={"entries": candidates})
    assert result.status_code == 409
    assert (root / names["old"] / "weights").read_bytes() == b"changed during cleanup"
    assert not list(root.glob(".cleanup-*"))
