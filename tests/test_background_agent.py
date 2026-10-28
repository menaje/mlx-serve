"""Authenticated background reconnects and ownership boundaries."""

import base64
import hashlib
import json
import os
import time

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from mlx_serve.core.background_agent import BackgroundControl, key_identifier
from mlx_serve.management_agent import create_agent

SECRET = "isolated-test-secret-" + "x" * 48
AAD = b"mlx-serve-agent-v1"
KEY_ID = key_identifier(SECRET)


def _key(secret):
    return hashlib.sha256(b"mlx-serve-agent-aes-key-v1:" + secret.encode()).digest()


def _seal(secret, message):
    nonce = os.urandom(12)
    cipher = AESGCM(_key(secret))
    data = json.dumps(message).encode()
    return nonce, base64.b64encode(nonce + cipher.encrypt(nonce, data, AAD)).decode()


def _open(secret, nonce, encoded):
    combined = base64.b64decode(encoded)
    cipher = AESGCM(_key(secret))
    data = cipher.decrypt(combined[:12], combined[12:], AAD)
    result = json.loads(data)
    assert base64.b64decode(result["request_nonce"]) == nonce
    return result


def test_secure_management_encrypts_both_directions_and_rejects_replay(tmp_path):
    app = create_agent(SECRET, data_dir=tmp_path)

    @app.post("/test-echo")
    async def echo(body: dict):
        return body

    local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1))
    remote = TestClient(app, base_url="http://localhost", client=("192.0.2.3", 1))
    message = {
        "timestamp": time.time(), "method": "POST", "path": "/test-echo",
        "body": {"token": "hf_sensitive_test_value"},
    }
    nonce, encoded = _seal(SECRET, message)
    assert "hf_sensitive_test_value" not in encoded
    response = local.post("/secure", json={"payload": encoded})
    assert response.status_code == 200
    assert "hf_sensitive_test_value" not in response.text
    result = _open(SECRET, nonce, response.json()["payload"])
    assert result["status"] == 200 and result["body"] == message["body"]
    assert local.post("/secure", json={"payload": encoded}).status_code == 403
    assert remote.post("/secure", json={"payload": encoded}).status_code == 403
    assert local.post(
        "/secure", headers={"Origin": "https://attacker.example"}, json={"payload": encoded},
    ).status_code == 403

    _, wrong = _seal("another-secret", message)
    assert local.post("/secure", json={"payload": wrong}).status_code == 403
    _, expired = _seal(SECRET, {**message, "timestamp": time.time() - 1000})
    assert local.post("/secure", json={"payload": expired}).status_code == 403
    _, nonfinite = _seal(SECRET, {**message, "timestamp": float("nan")})
    assert local.post("/secure", json={"payload": nonfinite}).status_code == 403
    _, recursive = _seal(SECRET, {**message, "path": "/secure"})
    assert local.post("/secure", json={"payload": recursive}).status_code == 403


def test_background_record_contains_no_secret_and_is_exclusive(tmp_path):
    first = BackgroundControl(tmp_path, 19001, os.getpid(), KEY_ID)
    second = BackgroundControl(tmp_path, 19002, os.getpid(), KEY_ID)
    first.enable()
    try:
        record = json.loads(first.record_path.read_text())
        assert record["pid"] == os.getpid() and record["port"] == 19001
        assert record["key_id"] == KEY_ID
        assert bytes.fromhex(record["key_id"]) != _key(SECRET)
        assert SECRET not in first.record_path.read_text()
        assert first.record_path.stat().st_mode & 0o077 == 0
        assert first.should_exit() is False
        with pytest.raises(ValueError, match="already running"):
            second.enable()
    finally:
        first.disable()
    assert not first.record_path.exists()


def test_stale_or_changed_record_cannot_delete_another_record(tmp_path, monkeypatch):
    control = BackgroundControl(tmp_path, 19001, os.getpid(), KEY_ID)
    control.enable()
    record = control.record_path
    contents = json.loads(record.read_text())
    contents["session"] = "another-session"
    record.write_text(json.dumps(contents))
    control.disable()
    assert record.exists()
    monkeypatch.setattr("mlx_serve.core.background_agent._process_identity", lambda pid: "alive")
    with pytest.raises(ValueError, match="may still be running"):
        BackgroundControl(tmp_path, 19002, os.getpid(), KEY_ID).enable()
    assert record.exists()


def test_background_rejects_symlinked_record_and_lock(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("do not overwrite")
    record = tmp_path / "app-agent.json"
    record.symlink_to(outside)
    with pytest.raises(ValueError, match="record is invalid"):
        BackgroundControl(tmp_path, 19001, os.getpid(), KEY_ID).enable()
    assert outside.read_text() == "do not overwrite"
    record.unlink()
    lock = tmp_path / ".app-agent.lock"
    lock.unlink()
    lock.symlink_to(outside)
    with pytest.raises(OSError):
        BackgroundControl(tmp_path, 19001, os.getpid(), KEY_ID).enable()
    assert outside.read_text() == "do not overwrite"


def test_reconnected_agent_stops_only_when_the_attached_app_is_gone(tmp_path, monkeypatch):
    import mlx_serve.core.background_agent as background

    control = BackgroundControl(tmp_path, 19001, os.getppid(), KEY_ID)
    monkeypatch.setattr(background, "_process_identity", lambda pid: "original-start")
    control.attach(os.getpid())
    assert not control.should_exit()
    control._last_activity -= 30
    monkeypatch.setattr(background, "_process_identity", lambda pid: "different-start")
    assert control.should_exit()
    monkeypatch.setattr(background, "_process_identity", lambda pid: None)
    assert not control.should_exit()
