"""Storage migration never removes the source or switches to an incomplete copy."""

import fcntl
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mlx_serve.core import storage_migration
from mlx_serve.core.default_models import default_model
from mlx_serve.core.managed_config import save_settings, settings_snapshot


@pytest.fixture
def storage(tmp_path, monkeypatch):
    source = tmp_path / "old" / "models"
    model = source / "sample"
    model.mkdir(parents=True)
    (model / "weights").write_bytes(b"verified model weights")
    original = source.parent / "decision-originals" / "sample-revision"
    original.mkdir(parents=True)
    (original / "source").write_bytes(b"original model")
    metadata = {
        "sample": {
            "type": "decision", "hf_repo": "example/model", "revision": "revision-a",
            "precision": "float16", "modified_at": "today", "original_path": str(original),
        }
    }
    (source.parent / "metadata.json").write_text(json.dumps(metadata))
    identity = {key: metadata["sample"].get(key) for key in (
        "type", "hf_repo", "revision", "precision", "modified_at",
    )}
    (source.parent / "default-models.json").write_text(json.dumps({
        "decision": {"name": "sample", "identity": identity},
    }))
    config = tmp_path / "config.yaml"
    monkeypatch.setattr("mlx_serve.core.managed_config._source_values", lambda: ({}, {}))
    save_settings({"models_dir": str(source)}, config)
    destination = tmp_path / "new" / "models"

    def save(changes, *, expected_models_dir):
        save_settings(changes, config, expected_models_dir=expected_models_dir)

    return SimpleNamespace(
        source=source, destination=destination, original=original, config=config, save=save,
    )


def _migrate(storage, *, cancelled=None, update=lambda *args: None):
    return storage_migration.migrate_storage(
        storage.source, storage.destination, cancelled=cancelled or threading.Event(),
        update=update, save=storage.save,
    )


def _effective_config(storage):
    return settings_snapshot(storage.config)["fields"]["models_dir"]["saved"]


def test_copy_preserves_original_and_updates_registry_defaults_and_settings(storage):
    preview = storage_migration.preview_storage_migration(storage.source, storage.destination)
    assert preview["size_bytes"] > 0 and preview["free_bytes"] > preview["size_bytes"]
    progress = []
    result = _migrate(storage, update=lambda *item: progress.append(item))
    target = storage.destination
    assert result["size_bytes"] == preview["size_bytes"]
    assert (target / "sample" / "weights").read_bytes() == b"verified model weights"
    assert (storage.source / "sample" / "weights").read_bytes() == b"verified model weights"
    assert (storage.original / "source").read_bytes() == b"original model"
    registry = json.loads((target.parent / "metadata.json").read_text())
    copied_original = target.parent / "decision-originals" / "sample-revision"
    assert registry["sample"]["original_path"] == str(copied_original)
    assert (copied_original / "source").read_bytes() == b"original model"
    assert default_model("decision", models_dir=target) == "sample"
    assert _effective_config(storage) == str(target)
    assert progress[-1][0] == "applying"


def test_migration_restores_read_only_directory_modes_after_copy(storage):
    model = storage.source / "sample"
    nested = model / "read-only"
    nested.mkdir()
    (nested / "weights").write_bytes(b"nested weights")
    nested.chmod(0o555)
    model.chmod(0o555)
    copied_model = storage.destination / "sample"
    try:
        _migrate(storage)
        copied_nested = copied_model / "read-only"
        assert (copied_nested / "weights").read_bytes() == b"nested weights"
        assert copied_model.stat().st_mode & 0o777 == 0o555
        assert copied_nested.stat().st_mode & 0o777 == 0o555
        assert model.stat().st_mode & 0o777 == 0o555
        assert _effective_config(storage) == str(storage.destination)
    finally:
        model.chmod(0o755)
        nested.chmod(0o755)
        if copied_model.exists():
            copied_model.chmod(0o755)
        copied_nested = copied_model / "read-only"
        if copied_nested.exists():
            copied_nested.chmod(0o755)


def test_failed_config_write_retains_source_and_the_copied_destination(storage):
    original_config = storage.config.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    storage.save = fail
    with pytest.raises(OSError, match="simulated"):
        _migrate(storage)
    assert storage.config.read_bytes() == original_config
    assert (storage.source / "sample" / "weights").read_bytes() == b"verified model weights"
    assert (storage.destination / "sample" / "weights").read_bytes() == b"verified model weights"


def test_changed_source_or_config_cannot_be_switched(storage, monkeypatch):
    original_copy = storage_migration._copy_file

    def modify_source(entry, target, cancelled, progress):
        original_copy(entry, target, cancelled, progress)
        if entry.source.name == "weights":
            entry.source.write_bytes(b"changed by another process")

    monkeypatch.setattr(storage_migration, "_copy_file", modify_source)
    with pytest.raises(ValueError, match="changed during migration"):
        _migrate(storage)
    assert _effective_config(storage) == str(storage.source)
    assert (storage.source / "sample" / "weights").read_bytes() == b"changed by another process"


def test_changed_copy_is_not_selected(storage, monkeypatch):
    original_copy = storage_migration._copy_file

    def modify_copy(entry, target, cancelled, progress):
        original_copy(entry, target, cancelled, progress)
        if entry.source.name == "source":
            (storage.destination / "sample" / "weights").write_bytes(b"changed destination")

    monkeypatch.setattr(storage_migration, "_copy_file", modify_copy)
    with pytest.raises(ValueError, match="Copied model files changed"):
        _migrate(storage)
    assert _effective_config(storage) == str(storage.source)
    assert (storage.source / "sample" / "weights").read_bytes() == b"verified model weights"


def test_configuration_compare_and_set_prevents_overwriting_a_concurrent_change(storage):
    another = storage.config.parent / "other" / "models"
    save_settings({"models_dir": str(another)}, storage.config)
    with pytest.raises(ValueError, match="settings changed"):
        _migrate(storage)
    assert _effective_config(storage) == str(another)
    assert (storage.source / "sample" / "weights").exists()
    assert (storage.destination / "sample" / "weights").exists()


def test_interruption_before_commit_leaves_old_configuration_and_files(storage):
    cancelled = threading.Event()

    def cancel_after_copy(phase, copied, total):
        if copied:
            cancelled.set()

    with pytest.raises(InterruptedError):
        _migrate(storage, cancelled=cancelled, update=cancel_after_copy)
    assert _effective_config(storage) == str(storage.source)
    assert (storage.source / "sample" / "weights").read_bytes() == b"verified model weights"


def test_existing_destination_and_links_are_never_overwritten(storage):
    storage.destination.parent.mkdir()
    marker = storage.destination.parent / "keep"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        _migrate(storage)
    assert marker.read_text() == "keep"
    marker.unlink()
    storage.destination.parent.rmdir()
    (storage.source / "sample" / "link").symlink_to(storage.source / "sample" / "weights")
    with pytest.raises(ValueError, match="link or another mounted volume"):
        _migrate(storage)
    assert not storage.destination.parent.exists()


def test_insufficient_space_and_active_preparation_leave_storage_unchanged(storage, monkeypatch):
    lock_dir = storage.source / ".decision-locks"
    lock_dir.mkdir()
    with (lock_dir / "d1-preparation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another process"):
            _migrate(storage)
        fcntl.flock(lock, fcntl.LOCK_UN)
    monkeypatch.setattr(storage_migration.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Not enough free disk"):
        _migrate(storage)
    assert not storage.destination.parent.exists()
    assert _effective_config(storage) == str(storage.source)


def test_environment_override_prevents_storage_switch(storage, monkeypatch):
    monkeypatch.setattr(
        "mlx_serve.core.managed_config._source_values",
        lambda: ({"models_dir": storage.source}, {}),
    )
    with pytest.raises(ValueError, match="overridden by the environment"):
        _migrate(storage)
    assert (storage.source / "sample" / "weights").exists()
    assert (storage.destination / "sample" / "weights").exists()


def test_agent_blocks_server_settings_and_jobs_during_migration(tmp_path, monkeypatch):
    import mlx_serve.management_agent as agent

    begun, finish = threading.Event(), threading.Event()
    source = tmp_path / "models"
    source.mkdir()
    monkeypatch.setattr(agent, "initialize_app_config", lambda: None)
    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(
        port=8000, host="127.0.0.1", models_dir=source,
    ))
    monkeypatch.setattr(
        agent, "settings_snapshot", lambda: {"fields": {"models_dir": {"source": "yaml"}}},
    )
    monkeypatch.setattr(agent, "_port_is_busy", lambda *args: False)

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    def migrate(*args, **kwargs):
        begun.set()
        assert finish.wait(5)
        return {"source": str(source), "destination": str(tmp_path / "new" / "models"),
                "size_bytes": 12}

    monkeypatch.setattr(agent, "_request_json", unavailable)
    monkeypatch.setattr(agent, "migrate_storage", migrate)
    app = agent.create_agent("private-secret", data_dir=tmp_path / "agent")
    header = {"Authorization": "Bearer private-secret"}
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1)) as client:
        started = client.post(
            "/storage/migration", headers=header,
            json={"destination": str(tmp_path / "new" / "models")},
        )
        assert started.json()["status"] == "running"
        assert begun.wait(3)
        assert client.post("/server/start", headers=header).status_code == 409
        changed = client.put("/settings", headers=header, json={"values": {"port": 9000}})
        assert changed.status_code == 409
        job = client.post("/jobs", headers=header, json={"repo": "example/model"})
        assert job.status_code == 409
        finish.set()
        for _ in range(100):
            status = client.get("/storage/migration", headers=header).json()
            if status["status"] != "running":
                break
            time.sleep(0.01)
        assert status["status"] == "succeeded"
