"""Encrypted local management requests for the app-owned agent."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_AAD = b"mlx-serve-agent-v1"
_MAX_MESSAGE_BYTES = 1024 * 1024
_MAX_CLOCK_SKEW = 120


class SecureAgentChannel:
    """Authenticate and encrypt requests using the per-launch secret."""

    def __init__(self, secret: str):
        material = b"mlx-serve-agent-aes-key-v1:" + secret.encode("utf-8")
        self._cipher = AESGCM(hashlib.sha256(material).digest())
        self._seen: dict[bytes, float] = {}

    def open(self, encoded: str) -> tuple[bytes, dict[str, Any]]:
        try:
            if not isinstance(encoded, str) or len(encoded) > 2 * _MAX_MESSAGE_BYTES:
                raise ValueError
            sealed = base64.b64decode(encoded, validate=True)
            if len(sealed) < 29 or len(sealed) > _MAX_MESSAGE_BYTES:
                raise ValueError
            nonce = sealed[:12]
            plaintext = self._cipher.decrypt(nonce, sealed[12:], _AAD)
            payload = json.loads(plaintext)
            timestamp = payload["timestamp"]
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError
            if abs(time.time() - timestamp) > _MAX_CLOCK_SKEW:
                raise ValueError
            now = time.monotonic()
            self._seen = {
                key: when for key, when in self._seen.items()
                if now - when < 2 * _MAX_CLOCK_SKEW
            }
            if nonce in self._seen or len(self._seen) >= 10000:
                raise ValueError
            self._seen[nonce] = now
            if not isinstance(payload, dict):
                raise ValueError
            return nonce, payload
        except (ValueError, TypeError, KeyError, InvalidTag, UnicodeDecodeError) as exc:
            raise ValueError("Secure management request is invalid") from exc

    def seal(self, nonce: bytes, status: int, body: Any) -> str:
        response = json.dumps({
            "request_nonce": base64.b64encode(nonce).decode("ascii"),
            "status": status, "body": body,
        }, separators=(",", ":")).encode("utf-8")
        response_nonce = os.urandom(12)
        combined = response_nonce + self._cipher.encrypt(response_nonce, response, _AAD)
        return base64.b64encode(combined).decode("ascii")
