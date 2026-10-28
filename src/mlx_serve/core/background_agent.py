"""Private discovery and encrypted reconnects for an app-owned background agent."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_AAD = b"mlx-serve-agent-v1"
_MAX_MESSAGE_BYTES = 1024 * 1024
_MAX_CLOCK_SKEW = 120


def key_identifier(secret: str) -> str:
    """Use a distinct derivation from the encryption key for public discovery."""
    return hashlib.sha256(b"mlx-serve-agent-key-id-v1:" + secret.encode("utf-8")).hexdigest()


def _process_identity(pid: int) -> str | None:
    if pid <= 1:
        return ""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return ""
    except PermissionError:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    if result.returncode == 1 and not value:
        return ""
    return value if result.returncode == 0 and value else None


class BackgroundControl:
    """Own a discovery record without ever taking control of a foreign process."""

    def __init__(self, data_dir: Path, port: int, parent_pid: int, key_id: str):
        self.data_dir = data_dir
        self.port = port
        self.parent_pid = parent_pid
        self.key_id = key_id
        self.enabled = False
        self.session = secrets.token_hex(16)
        self._lock_file = None
        self._attached: tuple[int, str] | None = None
        self._last_activity = time.monotonic()

    @property
    def record_path(self) -> Path:
        return self.data_dir / "app-agent.json"

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def attach(self, pid: int) -> None:
        identity = _process_identity(pid)
        if not identity:
            raise ValueError("The application process could not be verified")
        self._attached = (pid, identity)
        self.touch()

    def should_exit(self) -> bool:
        if self.enabled:
            return False
        if self._attached is None:
            return os.getppid() != self.parent_pid
        pid, identity = self._attached
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() - self._last_activity < 15:
            return False
        current = _process_identity(pid)
        return current is not None and current != identity

    def enable(self) -> None:
        if self.enabled:
            return
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.data_dir / ".app-agent.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        lock = os.fdopen(fd, "a+")
        try:
            info = os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ValueError("Background agent lock is invalid")
            os.fchmod(lock.fileno(), 0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another background agent is already running") from exc
            record = self.record_path
            if record.exists() or record.is_symlink():
                if record.is_symlink() or not record.is_file():
                    raise ValueError("Background agent record is invalid")
                try:
                    previous = json.loads(record.read_text(encoding="utf-8"))
                    pid = previous["pid"]
                    if previous.get("version") != 1 or not isinstance(pid, int) or pid <= 1:
                        raise ValueError
                except (KeyError, OSError, ValueError, TypeError) as exc:
                    raise ValueError("Background agent record is invalid") from exc
                if _process_identity(pid) != "":
                    raise ValueError("Another background agent may still be running")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.data_dir, delete=False
                ) as file:
                    temporary = file.name
                    os.fchmod(file.fileno(), 0o600)
                    json.dump({
                        "version": 1, "pid": os.getpid(), "port": self.port,
                        "session": self.session, "key_id": self.key_id,
                    }, file)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, record)
                temporary = None
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            self._lock_file = lock
            self.enabled = True
        except BaseException:
            lock.close()
            raise

    def disable(self) -> None:
        self.enabled = False
        if self._lock_file is None:
            return
        try:
            record = self.record_path
            if not record.is_symlink():
                try:
                    info = record.lstat()
                    data = json.loads(record.read_text(encoding="utf-8"))
                    if data.get("session") == self.session and record.lstat().st_ino == info.st_ino:
                        record.unlink()
                except (OSError, ValueError, AttributeError):
                    pass
        finally:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None


class SecureAgentChannel:
    """Authenticate and encrypt reconnect requests without exposing the Keychain secret."""

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
