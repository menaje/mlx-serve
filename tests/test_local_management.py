"""Tests for configuration, local-only access and durable preparation jobs."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlx_serve.config import Settings, settings
from mlx_serve.core.managed_config import initialize_app_config, save_settings, settings_snapshot
from mlx_serve.core.managed_jobs import JobManager, _error_guidance
from mlx_serve.server import create_app


@pytest.mark.parametrize(
    ("error", "category"),
    [
        ("401 Unauthorized", "authentication"),
        ("403 gated repository requires approval", "access_approval"),
        ("Permission denied", "storage_permission"),
        ("Insufficient free disk space", "disk"),
        ("Out of memory", "memory"),
        ("Network connection failed", "network"),
        ("Model file not found", "missing_file"),
        ("Model checksum mismatch", "validation"),
        ("Model conversion failed", "conversion"),
    ],
)
def test_preparation_errors_offer_relevant_next_steps(error, category):
    actual, guidance = _error_guidance(error)
    assert actual == category
    assert guidance


def test_settings_save_is_atomic_and_reports_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  port: 8000\nother:\n  keep: true\n")
    monkeypatch.setenv("MLX_SERVE_PORT", "9001")
    save_settings({"port": 9000, "models_dir": str(tmp_path / "new-models")}, path)
    data = settings_snapshot(path, applied=Settings.model_construct(port=8000))
    port = data["fields"]["port"]
    assert (port["saved"], port["effective_on_restart"], port["applied"]) == (9000, 9001, 8000)
    assert port["source"] == "environment" and port["restart_required"]
    before = path.read_text()
    assert "keep: true" in before
    with pytest.raises(ValidationError):
        save_settings({"port": 70000}, path)
    assert path.read_text() == before
    assert path.stat().st_mode & 0o077 == 0


def test_invalid_config_is_not_overwritten_or_included_in_errors(tmp_path):
    path = tmp_path / "config.yaml"
    original = "server: [sensitive-value\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="Configuration is not valid YAML") as error:
        save_settings({"port": 9000}, path)
    assert "sensitive-value" not in str(error.value)
    assert path.read_text() == original


def test_cli_config_parse_error_does_not_log_config_contents(tmp_path, caplog):
    from mlx_serve.core.config_loader import load_yaml_config

    path = tmp_path / "config.yaml"
    path.write_text("server: [hf_sensitive_config_value\n")
    assert load_yaml_config(path) == {}
    assert "Failed to parse config file" in caplog.text
    assert "hf_sensitive_config_value" not in caplog.text


def test_new_app_config_is_local_and_existing_cli_assets_are_preserved(tmp_path, monkeypatch):
    # Simulate an environment with no globally installed Homebrew CLI.
    original_exists = Path.exists

    def exists(path):
        if str(path) in ("/opt/homebrew/bin/mlx-serve", "/usr/local/bin/mlx-serve"):
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    fresh = tmp_path / "fresh" / "config.yaml"
    assert initialize_app_config(fresh)
    assert settings_snapshot(fresh)["fields"]["host"]["saved"] == "127.0.0.1"
    old = tmp_path / "old" / "config.yaml"
    old.parent.mkdir()
    (old.parent / "metadata.json").write_text("{}")
    assert not initialize_app_config(old)
    assert not old.exists()


def test_lan_and_browser_cannot_use_management_but_can_use_inference_metadata():
    app = create_app()
    remote = TestClient(app, base_url="http://127.0.0.1", client=("192.0.2.25", 1))
    local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    rebound = TestClient(app, base_url="http://attacker.example", client=("127.0.0.1", 1))
    assert remote.post("/api/pull", json={"name": "x"}).status_code == 403
    assert remote.get("/v1/models").status_code == 200
    assert rebound.get("/management/settings").status_code == 403
    response = local.post("/api/pull", json={"name": "x"}, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert local.get("/management/settings").status_code == 200


def test_app_server_requires_secret_even_on_loopback(monkeypatch):
    monkeypatch.setenv("MLX_SERVE_MANAGEMENT_TOKEN", "private-secret")
    app = create_app()
    local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    remote = TestClient(app, base_url="http://localhost", client=("192.0.2.4", 1))
    header = {"Authorization": "Bearer private-secret"}
    assert local.get("/management/settings").status_code == 403
    assert local.get("/management/settings", headers=header).status_code == 200
    assert remote.get("/management/settings", headers=header).status_code == 403
    assert local.get("/v1/models").status_code == 200


def test_management_agent_requires_secret_and_local_non_browser_requests(tmp_path, monkeypatch):
    from mlx_serve.management_agent import create_agent

    monkeypatch.setattr("mlx_serve.management_agent.initialize_app_config", lambda: None)
    app = create_agent("private-secret", data_dir=tmp_path)
    header = {"Authorization": "Bearer private-secret"}
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1)) as local:
        assert local.get("/jobs").status_code == 403
        assert local.get("/jobs", headers=header).status_code == 200
        secret = "hf_" + "s" * 5000
        invalid = local.post("/jobs", headers=header, json={"repo": "model", "token": secret})
        assert invalid.status_code == 422
        assert secret not in invalid.text
        browser = local.get("/jobs", headers={**header, "Origin": "https://evil.example"})
        assert browser.status_code == 403
        assert local.get("http://evil.example/jobs", headers=header).status_code == 403
    with TestClient(app, base_url="http://localhost", client=("192.0.2.4", 1)) as remote:
        assert remote.get("/jobs", headers=header).status_code == 403


def test_internal_http_bypasses_proxies_and_rejects_redirects(monkeypatch):
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from mlx_serve.core.local_access import open_local_http

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/stolen")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("http_proxy", url)
    monkeypatch.setenv("no_proxy", "")
    try:
        with open_local_http(urllib.request.Request(url + "/ok"), timeout=3) as response:
            assert response.read() == b"ok"
        localhost = f"http://localhost:{server.server_port}/ok"
        with open_local_http(urllib.request.Request(localhost), timeout=3) as response:
            assert response.read() == b"ok"
        request = urllib.request.Request(url + "/redirect", headers={"Authorization": "secret"})
        with pytest.raises(urllib.error.HTTPError) as error:
            open_local_http(request, timeout=3)
        assert error.value.code == 302
        assert requests == ["/ok", "/ok", "/redirect"]
        with pytest.raises(ValueError, match="loopback"):
            open_local_http(urllib.request.Request("http://192.0.2.1/"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_port_detection_distinguishes_a_listener_from_recently_closed_connections():
    import socket

    from mlx_serve.management_agent import _port_is_busy

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(2)
    port = server.getsockname()[1]
    assert _port_is_busy(port)
    probe, _ = server.accept()
    probe.close()
    client = socket.create_connection(("127.0.0.1", port))
    client.settimeout(2)
    connection, _ = server.accept()
    connection.close()
    assert client.recv(1) == b""
    client.close()
    server.close()
    assert not _port_is_busy(port)


def test_port_detection_finds_a_wildcard_listener():
    import socket

    from mlx_serve.management_agent import _port_is_busy

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 0))
        server.listen()
        assert _port_is_busy(server.getsockname()[1])


def test_port_detection_finds_a_reserved_wildcard_before_listen():
    import socket

    from mlx_serve.management_agent import _port_is_busy

    with socket.socket() as server:
        server.bind(("0.0.0.0", 0))
        assert _port_is_busy(server.getsockname()[1])


def test_storage_change_cannot_strand_existing_models_or_running_jobs(tmp_path):
    from mlx_serve.management_agent import _check_storage_change

    current = tmp_path / "models"
    current.mkdir()
    destination = tmp_path / "new" / "models"
    _check_storage_change(current, str(destination), active_jobs=False)
    with pytest.raises(ValueError, match="Finish or cancel"):
        _check_storage_change(current, str(destination), active_jobs=True)
    (current / "model").mkdir()
    with pytest.raises(ValueError, match="migrated"):
        _check_storage_change(current, str(destination), active_jobs=False)
    _check_storage_change(current, str(current), active_jobs=False)
    (current / "model").rmdir()
    (tmp_path / "metadata.json").write_text('{"model": {}}')
    with pytest.raises(ValueError, match="registered models"):
        _check_storage_change(current, str(destination), active_jobs=False)


def test_unmounted_volume_is_not_recreated_by_settings(tmp_path):
    from mlx_serve.core.storage_access import model_volume_mounted

    volume = Path("/Volumes") / f"mlx-serve-test-{tmp_path.name}"
    assert not volume.exists()
    directory = volume / "models"
    assert not model_volume_mounted(directory)
    with pytest.raises(OSError, match="volume is unavailable"):
        Settings(models_dir=directory).ensure_dirs()
    assert not volume.exists()
    config = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="volume is unavailable"):
        save_settings({"models_dir": str(directory)}, config)
    assert not config.exists()


def test_job_cleanup_waits_for_the_storage_volume(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as managed_jobs

    job_id = "b" * 32
    stage = tmp_path / f".decision-prepare-{job_id}-stage"
    stage.mkdir()
    monkeypatch.setattr(managed_jobs, "model_storage_available", lambda root: False)
    assert not JobManager._cleanup_stage(job_id, {"models_dir": str(tmp_path)})
    assert stage.exists()


def test_missing_storage_is_not_reported_as_an_empty_model_list(tmp_path, monkeypatch, client):
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr(settings, "models_dir", tmp_path / "detached" / "models")
    metadata = {"existing": {"type": "decision"}}
    monkeypatch.setattr(model_manager, "_metadata", metadata)
    called = []
    monkeypatch.setattr(model_manager, "_load_metadata", lambda: called.append(True))
    response = client.get("/management/state")
    assert response.status_code == 200
    state = response.json()
    assert state["state"] == "degraded"
    assert state["storage"]["state"] == "unavailable"
    assert state["models"] is None
    assert not called and model_manager._metadata == metadata
    rejected = client.post("/management/models/load", json={"name": "existing", "type": "decision"})
    assert rejected.status_code == 503


def test_agent_rejects_storage_change_when_models_are_present(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    models = tmp_path / "models"
    (models / "existing").mkdir(parents=True)
    monkeypatch.setattr(agent, "initialize_app_config", lambda: None)
    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(
        port=8000, host="127.0.0.1", models_dir=models,
    ))
    monkeypatch.setattr(agent, "_port_is_busy", lambda *args: False)

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    monkeypatch.setattr(agent, "_request_json", unavailable)
    written = []
    monkeypatch.setattr(agent, "save_settings", lambda values: written.append(values))
    app = agent.create_agent("private-secret", data_dir=tmp_path / "agent")
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1)) as client:
        response = client.put(
            "/settings", headers={"Authorization": "Bearer private-secret"},
            json={"values": {"models_dir": str(tmp_path / "new" / "models")}},
        )
    assert response.status_code == 422
    assert "migrated" in response.json()["detail"]
    assert not written
    assert (models / "existing").is_dir()


@pytest.mark.asyncio
async def test_external_server_or_port_conflict_is_never_stopped(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(port=8000, host="::1"))
    monkeypatch.setattr(
        agent, "_request_json",
        lambda *args, **kwargs: {"role": "gateway", "version": "1", "status": "healthy"},
    )
    controller = agent.ServerController(tmp_path)
    state = await controller.status()
    assert state["ownership"] == "external"
    assert state["url"] == "http://[::1]:8000"
    with pytest.raises(HTTPException) as error:
        await controller.stop()
    assert error.value.status_code == 409

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    monkeypatch.setattr(agent, "_request_json", unavailable)
    monkeypatch.setattr(agent, "_port_is_busy", lambda port, host: True)
    assert (await controller.status())["state"] == "port_conflict"
    with pytest.raises(HTTPException) as error:
        await controller.start()
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_unexpected_server_exit_is_visible_and_allows_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(
        port=8000, host="127.0.0.1", models_dir=tmp_path,
    ))
    monkeypatch.setattr(agent, "_port_is_busy", lambda *args: False)

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    monkeypatch.setattr(agent, "_request_json", unavailable)
    controller = agent.ServerController(tmp_path)
    controller.process = SimpleNamespace(poll=lambda: 17)
    controller.port = 8000
    failed = await controller.status()
    assert failed["state"] == "failed"
    assert failed["last_exit_code"] == 17
    assert failed["ownership"] is None
    log = tmp_path / "app-server.log"
    log.write_text("previous diagnostic output")
    log.chmod(0o644)
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(
        poll=lambda: None,
    ))
    restarted = await controller.start()
    assert restarted["state"] == "starting"
    assert restarted["ownership"] == "app"
    assert log.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_unresponsive_app_server_is_not_reported_as_starting_forever(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(port=8000, host="127.0.0.1"))
    controller = agent.ServerController(tmp_path)
    controller.process = SimpleNamespace(poll=lambda: None)
    controller.port = 8000
    controller.started_at = agent.time.monotonic()
    monkeypatch.setattr(
        agent, "_request_json",
        lambda *args, **kwargs: {"role": "gateway", "version": "1", "status": "healthy"},
    )
    assert (await controller.status())["state"] == "connected"

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    monkeypatch.setattr(agent, "_request_json", unavailable)
    assert (await controller.status())["state"] == "degraded"
    controller.ready_once = False
    controller.started_at -= 31
    assert (await controller.status())["state"] == "degraded"


def test_invalid_inference_does_not_log_the_request_body(caplog):
    client = TestClient(create_app(), base_url="http://localhost", client=("127.0.0.1", 1))
    secret = "hf_NOT_FOR_SERVER_LOGS_1234"
    response = client.post("/v1/decisions", json={"state": secret})
    assert response.status_code == 422
    assert secret not in caplog.text
    assert "Request validation failed" in caplog.text


def test_debug_body_redacts_credentials_and_omits_unparseable_input():
    from mlx_serve.server import _redact_debug_body

    body = b'{"token":"not-an-hf-token", "message":"hf_longsecret123 says hello"}'
    result = _redact_debug_body(body)
    assert "not-an-hf-token" not in result
    assert "hf_longsecret123" not in result
    assert "says hello" in result
    assert _redact_debug_body(b'{"token": "unclosed secret') == "[unparseable request body omitted]"


def test_management_deletion_uses_decision_worker_path_and_rejects_other_types(monkeypatch):
    from mlx_serve.core.model_manager import model_manager

    async def delete(payload, request):
        return {"status": "success", "name": payload.name}

    monkeypatch.setattr(model_manager, "_load_metadata", lambda: None)
    monkeypatch.setattr(model_manager, "get_model_type", lambda name: "decision")
    monkeypatch.setattr("mlx_serve.routers.models.delete_model", delete)
    client = TestClient(create_app(), base_url="http://localhost", client=("127.0.0.1", 1))
    response = client.post("/management/models/delete", json={"name": "d1", "type": "decision"})
    assert response.json() == {"status": "success", "name": "d1"}
    assert client.post(
        "/management/models/delete", json={"name": "other", "type": "embedding"}
    ).status_code == 501


@pytest.mark.asyncio
async def test_job_cancel_terminates_process_and_preserves_existing_models(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    existing = models / "existing"
    existing.mkdir()
    (existing / "model").write_text("keep")
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, os, pathlib, signal, sys, time\n"
        "p=json.loads(sys.stdin.readline())\n"
        "job=sys.argv[1]\n"
        "stage=pathlib.Path(p['models_dir']) / ('.decision-prepare-'+job+'-fake')\n"
        "stage.mkdir()\n"
        "print(json.dumps({'status':'downloading'}), flush=True)\n"
        "print(json.dumps({'status':'resources','temporary_file_bytes':42}), flush=True)\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))\n"
        "while True: time.sleep(.1)\n"
    )
    original = asyncio.create_subprocess_exec

    async def spawn_fake(*args, **kwargs):
        job_id = args[args.index("--job-id") + 1]
        return await original(sys.executable, str(script), job_id, **kwargs)

    monkeypatch.setattr(managed_jobs.asyncio, "create_subprocess_exec", spawn_fake)
    manager = JobManager(tmp_path / "jobs")
    await manager.recover()
    params = {"repo": "LiquidAI/d1-omni-600M", "models_dir": str(models)}
    job = manager.start(params, token="hf_secret123")
    with pytest.raises(ValueError, match="active job"):
        manager.start({**params, "name": "another-name"})
    for _ in range(100):
        current = manager.get(job["id"])
        if current and any(e.get("temporary_file_bytes") == 42 for e in current["events"]):
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("Job did not begin downloading")
    assert current["stage"] == "downloading"
    pid = current["pid"]
    result = await manager.cancel(job["id"])
    assert result["status"] == "cancelled"
    assert not list(models.glob(".decision-prepare-*"))
    assert (existing / "model").read_text() == "keep"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await manager.close()
    reopened = JobManager(tmp_path / "jobs")
    await reopened.recover()
    assert reopened.get(job["id"])["status"] == "cancelled"
    database_contents = (tmp_path / "jobs" / "jobs.sqlite3").read_bytes().decode(errors="ignore")
    assert "hf_secret123" not in database_contents
    await reopened.close()


@pytest.mark.asyncio
async def test_recovery_blocks_retry_until_process_and_cleanup_are_confirmed(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    job_id = "a" * 32
    stage = models / f".decision-prepare-{job_id}-stage"
    stage.mkdir()
    (stage / "partial").write_text("unfinished")
    params = {
        "repo": "LiquidAI/d1-omni-600M", "name": None, "revision": None,
        "keep_original": False, "models_dir": str(models),
    }
    key = json.dumps({k: v for k, v in params.items() if k != "name"}, sort_keys=True)
    manager = JobManager(tmp_path / "jobs")
    with manager._db() as db:
        db.execute(
            "INSERT INTO jobs (id, request_key, params, status, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', 'downloading', 'now', 'now')",
            (job_id, key, json.dumps(params)),
        )
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: None)
    await manager.recover()
    assert manager.get(job_id)["status"] == "recovery_pending"
    assert stage.exists()
    with pytest.raises(ValueError, match="active job"):
        manager.start(params)
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: [])
    await manager.recover()
    assert manager.get(job_id)["status"] == "interrupted"
    assert not stage.exists()
    await manager.close()
