"""Authenticated and encrypted local management requests."""

import base64
import hashlib
import json
import os
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from mlx_serve.management_agent import create_agent

SECRET = "isolated-test-secret-" + "x" * 48
AAD = b"mlx-serve-agent-v1"


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
