"""Tests for configuration, local-only access and durable preparation jobs."""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
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


def test_optional_queue_and_memory_limits_round_trip_and_reject_invalid_values(tmp_path):
    path = tmp_path / "config.yaml"
    values = {
        "models_dir": str(tmp_path / "models"),
        "inference_max_queue_per_model": None,
        "inference_queue_timeout_seconds": None,
        "memory_process_limit_fraction": None,
        "memory_min_available_fraction": 0.15,
        "memory_load_headroom_fraction": 0.0,
        "app_log_retention_days": None,
    }
    save_settings(values, path)
    snapshot = settings_snapshot(path)["fields"]
    assert {key: snapshot[key]["saved"] for key in values} == values

    save_settings({"inference_max_queue_per_model": 0,
                   "inference_queue_timeout_seconds": 12.5,
                   "app_log_retention_days": 30}, path)
    snapshot = settings_snapshot(path)["fields"]
    assert snapshot["inference_max_queue_per_model"]["saved"] == 0
    assert snapshot["inference_queue_timeout_seconds"]["saved"] == 12.5
    assert snapshot["app_log_retention_days"]["saved"] == 30
    before = path.read_text()
    for changes in (
        {"inference_queue_timeout_seconds": 0},
        {"memory_process_limit_fraction": 1.0},
        {"memory_load_headroom_fraction": -0.1},
        {"app_log_retention_days": 0},
        {"app_log_retention_days": 36501},
    ):
        with pytest.raises(ValidationError):
            save_settings(changes, path)
        assert path.read_text() == before


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


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (0, "state = running", "", "running"),
        (0, "state = waiting", "", "loaded"),
        (113, "", "Could not find service", "not_loaded"),
        (1, "", "permission denied", "unknown"),
    ],
)
def test_existing_launchd_service_is_only_inspected(
    tmp_path, monkeypatch, returncode, stdout, stderr, expected,
):
    from mlx_serve import management_agent

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(management_agent.sys, "platform", "darwin")
    monkeypatch.setattr(management_agent.shutil, "which", lambda _: "/example/mlx-serve")
    plist = tmp_path / "Library/LaunchAgents/com.mlx-serve.server.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("existing service")
    commands = []

    def inspect(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(management_agent.subprocess, "run", inspect)
    snapshot = management_agent._installation_snapshot()
    assert commands == [[
        "/bin/launchctl", "print", f"gui/{os.getuid()}/com.mlx-serve.server",
    ]]
    assert snapshot["launchd"]["user_state"] == expected
    assert str(plist) in snapshot["launchd"]["files"]
    assert snapshot["cli_in_path"] == "/example/mlx-serve"
    assert plist.read_text() == "existing service"


def test_launchd_inspection_timeout_reports_unknown(tmp_path, monkeypatch):
    from mlx_serve import management_agent

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(management_agent.sys, "platform", "darwin")

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 2)

    monkeypatch.setattr(management_agent.subprocess, "run", timeout)
    assert management_agent._installation_snapshot()["launchd"]["user_state"] == "unknown"


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
        invalid = local.post("/jobs", headers=header, json={
            "repo": "LiquidAI/d1-omni-600M", "token": "hf_test_secret",
        })
        assert invalid.status_code == 422
        assert "hf_test_secret" not in invalid.text
        assert local.post("/background", headers=header, json={"enabled": True}).status_code == 404
        assert local.post(
            "/background/attach", headers=header, json={"pid": 123},
        ).status_code == 404
        browser = local.get("/jobs", headers={**header, "Origin": "https://evil.example"})
        assert browser.status_code == 403
        assert local.get("http://evil.example/jobs", headers=header).status_code == 403
    with TestClient(app, base_url="http://localhost", client=("192.0.2.4", 1)) as remote:
        assert remote.get("/jobs", headers=header).status_code == 403


def test_diagnostics_excludes_job_inputs_errors_and_untrusted_health(tmp_path, monkeypatch):
    from mlx_serve.management_agent import ServerController, create_agent

    async def status(_controller):
        return {
            "state": "connected", "ownership": "external", "port": 8000,
            "health": {"status": "healthy", "extra": "health_secret"},
        }

    monkeypatch.setattr("mlx_serve.management_agent.initialize_app_config", lambda: None)
    monkeypatch.setattr(ServerController, "status", status)
    app = create_agent("private-secret", data_dir=tmp_path)
    header = {"Authorization": "Bearer private-secret"}
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1)) as client:
        params = {"revision": "hf_revision_secret", "name": "private_model_input"}
        with sqlite3.connect(tmp_path / "jobs" / "jobs.sqlite3") as db:
            db.execute(
                "INSERT INTO jobs (id, request_key, params, status, stage, created_at, "
                "updated_at, result, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "a" * 32, "fixture", json.dumps(params), "failed", "failed",
                    "2026-10-28T00:00:00+00:00", "2026-10-28T00:01:00+00:00",
                    json.dumps({"detail": "result_secret"}),
                    "Network connection failed: error_secret",
                ),
            )
        response = client.get("/diagnostics", headers=header)
        assert response.status_code == 200
        for secret in (
            "hf_revision_secret", "private_model_input", "result_secret",
            "error_secret", "health_secret",
        ):
            assert secret not in response.text
        report = response.json()
        assert report["server"]["state"] == "connected"
        assert report["jobs"][0]["status"] == "failed"
        assert report["jobs"][0]["error_category"] == "network"


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


@pytest.mark.parametrize(
    ("repo", "architecture", "precision", "required", "validation"),
    [
        ("LiquidAI/d1-omni-600M", "d1_omni", "float16", ["LICENSE", "model.safetensors"],
         {"official_predict": True, "max_probability_difference": 0.001, "tolerance": 0.02}),
        ("vllm-sr/d3-lite", "qwen3_5", "bfloat16",
         ["decision_config.json", "readout.safetensors", "model.safetensors"],
         {"mlx_predict": True, "published_file_hashes": True}),
    ],
)
def test_management_reports_recorded_preparation_without_rehashing(
    tmp_path, monkeypatch, client, repo, architecture, precision, required, validation,
):
    from mlx_serve.core import decision_prepare
    from mlx_serve.core.decision_prepare_worker import MANIFEST
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    for filename in ["config.json", "tokenizer.json", *required]:
        (model_dir / filename).write_text("fixture")
    revision = "a" * 40
    manifest = {
        "schema_version": 1, "source_repo": repo, "revision": revision,
        "architecture": architecture, "precision": precision,
        "files_sha256": {item.name: "recorded-hash" for item in model_dir.iterdir()},
        "license": "test-license", "converted": True, "converter": "test-converter",
        "converter_version": "1.0", "converter_options": {"dtype": precision},
        "validation": {**validation, "sample_answers": ["private sample"]},
    }
    (model_dir / MANIFEST).write_text(json.dumps(manifest))
    metadata = {"test-model": {"type": "decision", "hf_repo": repo, "revision": revision}}
    monkeypatch.setattr(model_manager, "_metadata", metadata)
    monkeypatch.setattr(model_manager, "_load_metadata", lambda: None)

    def unexpected_hash(_):
        raise AssertionError("Refreshing the UI must not rehash model weights")

    monkeypatch.setattr(decision_prepare, "file_hash", unexpected_hash)
    response = client.get("/management/state")
    assert response.status_code == 200
    record = response.json()["models"][0]["preparation"]
    assert record["state"] == "recorded"
    assert record["license"] == "test-license"
    assert record["converter_options"] == {"dtype": precision}
    assert record["validation"] == validation
    assert "private sample" not in response.text

    metadata["test-model"]["revision"] = "b" * 40
    assert client.get("/management/state").json()["models"][0]["preparation"] == {
        "state": "unavailable"
    }
    metadata["test-model"].pop("revision")
    assert client.get("/management/state").json()["models"][0]["preparation"] == {
        "state": "unavailable"
    }


def test_existing_mlx_record_does_not_claim_a_local_converter(tmp_path, monkeypatch):
    from mlx_serve.core.decision_prepare_worker import MLX_REPO
    from mlx_serve.routers import management

    manifest = {
        "source_repo": MLX_REPO, "revision": "a" * 40,
        "converted": False, "converter": "mlx-vlm", "converter_version": "0.7.10",
        "validation": {"official_predict": True},
    }
    monkeypatch.setattr(management, "read_verified_manifest", lambda *_args, **_kwargs: manifest)
    record = management._preparation_record(
        tmp_path, {"revision": "a" * 40, "hf_repo": MLX_REPO}, "decision",
    )
    assert record["converted"] is False
    assert record["converter"] is None and record["converter_version"] is None
    assert manifest["converter"] == "mlx-vlm"  # The existing recorded file is not rewritten.


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
async def test_automatic_load_ignores_proxies_and_does_not_follow_redirects(monkeypatch):
    import httpx

    import mlx_serve.management_agent as agent

    requests = []

    async def respond(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await agent._post_local_management(
                port, "/management/models/load", {"name": "d1", "type": "decision"},
                authorization="local-secret", host="127.0.0.1",
            )
        assert len(requests) == 1
        assert b"authorization: bearer local-secret" in requests[0].lower()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unexpected_server_exit_is_visible_and_allows_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(
        port=8000, host="127.0.0.1", models_dir=tmp_path, app_log_retention_days=None,
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
    monkeypatch.setenv("HF_TOKEN", "hf_environment_fixture")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_fixture")

    def spawn_server(*args, **kwargs):
        environment = kwargs["env"]
        assert "HF_TOKEN" not in environment and "HUGGING_FACE_HUB_TOKEN" not in environment
        assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert environment["MLX_SERVE_MANAGEMENT_TOKEN"]
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(agent.subprocess, "Popen", spawn_server)
    restarted = await controller.start()
    assert restarted["state"] == "starting"
    assert restarted["ownership"] == "app"
    assert log.read_text() == "previous diagnostic output"
    created_logs = list((tmp_path / "app-server-logs").glob("running-*.log"))
    assert len(created_logs) == 1 and created_logs[0].stat().st_mode & 0o077 == 0


def test_server_log_retention_preserves_active_legacy_and_unrelated_files(tmp_path):
    from mlx_serve.management_agent import _clean_completed_server_logs

    old = tmp_path / f"server-{'a' * 32}.log"
    recent = tmp_path / f"server-{'b' * 32}.log"
    active = tmp_path / f"running-{'c' * 32}.log"
    unrelated = tmp_path / "app-server.log"
    linked = tmp_path / f"server-{'d' * 32}.log"
    hard_linked = tmp_path / f"server-{'e' * 32}.log"
    for path in (old, recent, active, unrelated):
        path.write_text(path.name)
    linked.symlink_to(unrelated)
    os.link(unrelated, hard_linked)
    expired = time.time() - 10 * 24 * 60 * 60
    for path in (old, active, unrelated):
        os.utime(path, (expired, expired))
    assert _clean_completed_server_logs(tmp_path, None) == 0
    assert old.exists()
    assert _clean_completed_server_logs(tmp_path, 7) == 0
    assert not old.exists()
    assert recent.exists() and active.exists() and unrelated.exists() and linked.is_symlink()
    assert hard_linked.exists()


@pytest.mark.parametrize("kind", ["file", "file_link", "directory_link"])
@pytest.mark.asyncio
async def test_server_log_directory_rejects_links_and_files(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace

    from fastapi import HTTPException

    import mlx_serve.management_agent as agent

    target = tmp_path / "user-data"
    if kind == "directory_link":
        target.mkdir()
        content = target / "data"
    else:
        content = target
    content.write_text("keep")
    log_dir = tmp_path / "app-server-logs"
    if kind != "file":
        log_dir.symlink_to(target)
    else:
        log_dir.write_text("keep")
    monkeypatch.setattr(agent, "Settings", lambda: SimpleNamespace(
        port=8000, host="127.0.0.1", models_dir=tmp_path, app_log_retention_days=7,
    ))
    monkeypatch.setattr(agent, "_port_is_busy", lambda *args: False)

    def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503)

    monkeypatch.setattr(agent, "_request_json", unavailable)
    with pytest.raises(HTTPException, match="Server log directory is unavailable"):
        await agent.ServerController(tmp_path).start()
    assert content.read_text() == "keep"
    assert log_dir.is_symlink() if kind != "file" else log_dir.read_text() == "keep"


def test_server_log_is_completed_only_after_process_group_exits(tmp_path):
    from mlx_serve.management_agent import ServerController

    path = tmp_path / f"running-{'a' * 32}.log"
    path.write_text("server log")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
    )
    controller = ServerController(tmp_path)
    controller.process = process
    controller.log_path = path
    try:
        controller._finish_log()
        assert path.exists()
        process.terminate()
        process.wait(timeout=5)
        controller._finish_log()
        assert not path.exists()
        assert (tmp_path / f"server-{'a' * 32}.log").read_text() == "server log"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_reaped_server_group_is_not_signalled(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mlx_serve.management_agent import ServerController

    controller = ServerController(tmp_path)
    process = SimpleNamespace(pid=12345, returncode=-9)
    controller.process = process
    controller.process_group = process.pid
    signals = []
    monkeypatch.setattr("mlx_serve.management_agent.os.killpg", lambda *args: signals.append(args))
    controller._kill_owned_group(process)
    assert signals == []


@pytest.mark.parametrize("owns_group", [True, False])
def test_orphaned_job_guard_only_signals_a_group_it_owns(monkeypatch, owns_group):
    import mlx_serve.core.managed_job_runner as runner

    sent = []
    monkeypatch.setattr(runner.os, "getpid", lambda: 123)
    monkeypatch.setattr(runner.os, "getppid", lambda: 1)
    monkeypatch.setattr(runner.os, "getpgrp", lambda: 123 if owns_group else 456)
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: sent.append(("process", pid, sig)))
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: sent.append(("group", pid, sig)))

    class NeverFinished:
        def wait(self, _seconds):
            return False

    runner._enforce_parent_lifetime(999, NeverFinished())
    kind = "group" if owns_group else "process"
    assert sent == [(kind, 123, signal.SIGTERM), (kind, 123, signal.SIGKILL)]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["prompt", "messages"])
async def test_legacy_completion_does_not_log_inputs(field, monkeypatch, caplog):
    from fastapi import HTTPException

    from mlx_serve.routers.chat import CompletionRequest, create_completion

    secret = "private_completion_input_1234"
    value = secret if field == "prompt" else [{"role": "user", "content": secret}]
    request = CompletionRequest(model="fixture", user=secret, **{field: value})

    def stop_before_loading(*_args):
        raise HTTPException(status_code=400, detail="Fixture stopped before model loading")

    monkeypatch.setattr(
        "mlx_serve.routers.chat._ensure_generation_worker_kind", stop_before_loading
    )
    caplog.set_level(logging.INFO, logger="mlx_serve.routers.chat")
    with pytest.raises(HTTPException):
        await create_completion(request)
    assert secret not in caplog.text


def test_chat_template_failure_does_not_log_input(caplog):
    from mlx_serve.routers.chat import ChatMessage, _format_messages_for_model

    secret = "private_chat_template_input_1234"

    class FailingTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            raise ValueError(messages[0]["content"])

    prompt = _format_messages_for_model(
        [ChatMessage(role="user", content=secret)], FailingTokenizer()
    )

    assert secret in prompt
    assert "Chat template failed" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_batch_failure_does_not_log_input(caplog):
    from mlx_serve.core.batch_processor import BatchProcessor

    secret = "private_batch_input_1234"

    def fail_with_input(inputs):
        raise ValueError(inputs[0])

    processor = BatchProcessor(fail_with_input, max_wait_ms=1)
    try:
        with pytest.raises(ValueError, match=secret):
            await processor.submit(secret)
    finally:
        await processor.stop()

    assert "Batch processing failed" in caplog.text
    assert secret not in caplog.text


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
@pytest.mark.parametrize("repo", [
    "LiquidAI/d1-omni-600M", "vllm-sr/d3-lite",
    "mlx-community/all-MiniLM-L6-v2-bf16", "mlx-community/Qwen3-Reranker-0.6B-4bit",
])
@pytest.mark.parametrize("shutdown", [False, True])
async def test_job_cancel_terminates_process_and_preserves_existing_models(
    tmp_path, monkeypatch, repo, shutdown
):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    existing = models / "existing"
    existing.mkdir()
    (existing / "model").write_text("keep")
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    (hf_home / "token").write_text("hf_saved_fixture")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_TOKEN", "hf_secret123")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_other_fixture")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, os, pathlib, signal, sys, time\n"
        "from huggingface_hub.utils import build_hf_headers\n"
        "assert 'HF_TOKEN' not in os.environ and 'HUGGING_FACE_HUB_TOKEN' not in os.environ\n"
        "assert os.environ.get('HF_HUB_DISABLE_IMPLICIT_TOKEN') == '1'\n"
        "assert 'authorization' not in build_hf_headers()\n"
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
    params = {"repo": repo, "models_dir": str(models)}
    job = manager.start(params)
    with pytest.raises(ValueError, match="active job"):
        manager.start({**params, "name": "another-name"})
    with pytest.raises(ValueError, match="active job"):
        manager.start({**params, "load_after": True})
    for _ in range(100):
        current = manager.get(job["id"])
        if current and any(e.get("temporary_file_bytes") == 42 for e in current["events"]):
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("Job did not begin downloading")
    assert current["stage"] == "downloading"
    pid = current["pid"]
    if shutdown:
        await manager.close()
        result = manager.get(job["id"])
    else:
        result = await manager.cancel(job["id"])
    assert result["status"] == "cancelled"
    assert not list(models.glob(".decision-prepare-*"))
    assert (existing / "model").read_text() == "keep"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    if not shutdown:
        await manager.close()
    reopened = JobManager(tmp_path / "jobs")
    await reopened.recover()
    assert reopened.get(job["id"])["status"] == "cancelled"
    database_contents = (tmp_path / "jobs" / "jobs.sqlite3").read_bytes().decode(errors="ignore")
    assert "hf_secret123" not in database_contents
    await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("load_fails", [False, True])
async def test_preparation_records_automatic_loading_separately(tmp_path, monkeypatch, load_fails):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, sys\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'status':'success','name':'d1-test'}), flush=True)\n"
    )
    original = asyncio.create_subprocess_exec

    async def spawn_fake(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)

    calls = []

    async def load_model(name, kind):
        assert kind == "decision"
        calls.append(name)
        if load_fails:
            raise RuntimeError("unavailable")

    monkeypatch.setattr(managed_jobs.asyncio, "create_subprocess_exec", spawn_fake)
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: [])
    manager = JobManager(tmp_path / "jobs", load_model=load_model)
    job = manager.start({
        "repo": "LiquidAI/d1-omni-600M", "models_dir": str(models), "load_after": True,
    })
    await manager._tasks[job["id"]]
    completed = manager.get(job["id"])
    assert completed["status"] == "succeeded"
    assert completed["result"]["auto_load"] == ("failed" if load_fails else "succeeded")
    assert calls == ["d1-test"]
    assert any(
        event["status"] == ("load_failed" if load_fails else "load_succeeded")
        for event in completed["events"]
    )
    await manager.close()


@pytest.mark.asyncio
async def test_cancelling_during_automatic_load_does_not_delay_shutdown(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, sys\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'status':'success','name':'d1-test'}), flush=True)\n"
    )
    original = asyncio.create_subprocess_exec

    async def spawn_fake(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)

    entered = asyncio.Event()

    async def load_model(name, kind):
        assert kind == "decision"
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(managed_jobs.asyncio, "create_subprocess_exec", spawn_fake)
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: [])
    manager = JobManager(tmp_path / "jobs", load_model=load_model)
    job = manager.start({
        "repo": "LiquidAI/d1-omni-600M", "models_dir": str(models), "load_after": True,
    })
    await asyncio.wait_for(entered.wait(), timeout=5)
    completed = await asyncio.wait_for(manager.cancel(job["id"]), timeout=2)
    assert completed["status"] == "succeeded"
    assert completed["result"]["auto_load"] == "failed"
    await manager.close()


def test_agent_loads_a_prepared_model_without_an_open_app(tmp_path, monkeypatch):
    import time

    import mlx_serve.core.managed_jobs as managed_jobs
    import mlx_serve.management_agent as agent

    models = tmp_path / "models"
    models.mkdir()
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, sys\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'status':'success','name':'d1-test'}), flush=True)\n"
    )
    original = asyncio.create_subprocess_exec

    async def spawn_fake(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)

    async def server_status(self):
        return {"state": "connected", "ownership": "external", "port": 19001}

    calls = []

    def server_request(port, path, payload=None, **kwargs):
        assert path == "/management/settings"
        return {"fields": {"models_dir": {"applied": str(models)}}}

    async def load_request(port, path, payload, **kwargs):
        assert path == "/management/models/load"
        calls.append((port, payload))

    monkeypatch.setattr(agent, "initialize_app_config", lambda: None)
    monkeypatch.setattr(agent.ServerController, "status", server_status)
    monkeypatch.setattr(agent.ServerController, "address", lambda self, port: "127.0.0.1")
    monkeypatch.setattr(agent, "_request_json", server_request)
    monkeypatch.setattr(agent, "_post_local_management", load_request)
    monkeypatch.setattr(managed_jobs.asyncio, "create_subprocess_exec", spawn_fake)
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: [])
    headers = {"Authorization": "Bearer test-secret"}
    app = agent.create_agent("test-secret", data_dir=tmp_path)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1)) as client:
        started = client.post(
            "/jobs", headers=headers,
            json={"repo": "LiquidAI/d1-omni-600M", "load_after": True},
        )
        assert started.status_code == 200
        job_id = started.json()["id"]
        for _ in range(100):
            completed = client.get(f"/jobs/{job_id}", headers=headers).json()
            if completed["status"] == "succeeded":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Preparation did not finish")
        assert completed["result"]["auto_load"] == "succeeded"
        assert calls == [(19001, {"name": "d1-test", "type": "decision"})]


@pytest.mark.asyncio
async def test_recovery_finishes_a_persisted_automatic_load(tmp_path, monkeypatch):
    import mlx_serve.core.managed_jobs as managed_jobs

    models = tmp_path / "models"
    models.mkdir()
    params = {
        "repo": "LiquidAI/d1-omni-600M", "models_dir": str(models), "load_after": True,
    }
    calls = []

    async def load_model(name, kind):
        assert kind == "decision"
        calls.append(name)
        await asyncio.sleep(0.01)

    manager = JobManager(tmp_path / "jobs", load_model=load_model)
    with manager._db() as db:
        db.execute(
            "INSERT INTO jobs (id, request_key, params, status, stage, created_at, updated_at, "
            "result) VALUES (?, ?, ?, 'running', 'loading', 'now', 'now', ?)",
            ("a" * 32, "pending-load", json.dumps(params),
             json.dumps({"status": "success", "name": "d1-test"})),
        )
    monkeypatch.setattr(managed_jobs, "_job_worker_pids", lambda _: [])
    await asyncio.gather(manager.recover(), manager.recover())
    job = manager.get("a" * 32)
    assert job["status"] == "succeeded" and job["result"]["auto_load"] == "succeeded"
    assert calls == ["d1-test"]
    await manager.recover()
    assert calls == ["d1-test"]
    await manager.close()


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
