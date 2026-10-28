"""App-owned, authenticated loopback agent for server and model operations."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_serve.config import Settings
from mlx_serve.core.agent_channel import SecureAgentChannel
from mlx_serve.core.local_access import is_local_management_request, open_local_http
from mlx_serve.core.managed_config import (
    initialize_app_config,
    save_settings,
    settings_snapshot,
)
from mlx_serve.core.managed_jobs import ACTIVE_STATES, JobManager, public_hf_environment
from mlx_serve.core.storage_access import model_storage_available, model_volume_mounted
from mlx_serve.core.storage_migration import migrate_storage, preview_storage_migration

DATA_DIR = Path.home() / ".mlx-serve"


def _loopback_host(bind_host: str) -> str:
    return "::1" if bind_host in ("::1", "::") else "127.0.0.1"


def _server_url(port: int, host: str = "127.0.0.1") -> str:
    address = f"[{host}]" if ":" in host else host
    return f"http://{address}:{port}"


def _request_json(
    port: int, path: str, payload: Any = None, *, timeout: float = 3,
    authorization: str | None = None, host: str = "127.0.0.1",
) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = f"Bearer {authorization}"
    request = urllib.request.Request(
        _server_url(port, host) + path, data=body, headers=headers,
        method="GET" if body is None else "POST",
    )
    try:
        with open_local_http(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            message = json.load(exc).get("detail", "Server request failed")
        except (ValueError, AttributeError):
            message = "Server request failed"
        raise HTTPException(status_code=exc.code, detail=message) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Server is unavailable") from exc


async def _post_local_management(
    port: int, path: str, payload: dict[str, Any], *,
    authorization: str | None, host: str,
) -> None:
    headers = {"Authorization": f"Bearer {authorization}"} if authorization else {}
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=180) as client:
        response = await client.post(_server_url(port, host) + path, json=payload, headers=headers)
        response.raise_for_status()


def _port_is_busy(port: int, host: str = "127.0.0.1") -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    # On macOS a reuse-enabled bind to a specific address can succeed even
    # while another process listens on the wildcard address for this port.
    with socket.socket(family, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        try:
            if connection.connect_ex((host, port)) == 0:
                return True
        except OSError:
            pass
    # A process may reserve the wildcard address before it starts listening.
    # macOS can still allow a reuse-enabled bind to the specific loopback
    # address in that case, so also probe the wildcard address. This is
    # deliberately conservative if another interface owns the same port.
    for address in (host, "::" if family == socket.AF_INET6 else "0.0.0.0"):
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            # Match the server's socket behavior so TIME_WAIT is not a conflict.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((address, port))
            except OSError:
                return True
    return False


def _installation_snapshot() -> dict[str, Any]:
    """Inspect known CLI and launchd locations without changing a service."""
    home = Path.home()
    service_paths = [
        home / "Library/LaunchAgents/com.mlx-serve.server.plist",
        Path("/Library/LaunchAgents/com.mlx-serve.server.plist"),
        Path("/Library/LaunchDaemons/com.mlx-serve.server.plist"),
    ]
    launchd_state = "unknown"
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/bin/launchctl", "print", f"gui/{os.getuid()}/com.mlx-serve.server"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.returncode == 0:
                launchd_state = "running" if "state = running" in result.stdout else "loaded"
            elif result.returncode == 113 and "Could not find service" in result.stderr:
                launchd_state = "not_loaded"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "config_path": str(home / ".mlx-serve" / "config.yaml"),
        "homebrew_cli": next((p for p in (
            "/opt/homebrew/bin/mlx-serve", "/usr/local/bin/mlx-serve"
        ) if Path(p).exists()), None),
        "cli_in_path": shutil.which("mlx-serve"),
        "launchd": {
            "files": [str(path) for path in service_paths if path.exists()],
            "user_state": launchd_state,
        },
    }


def _check_storage_change(current: Path, requested: str, active_jobs: bool) -> None:
    """Do not strand existing models when the registry follows the storage path."""
    destination = Path(requested).expanduser()
    if current.resolve() == destination.resolve():
        return
    if active_jobs:
        raise ValueError("Finish or cancel model preparation before changing storage")
    if not model_volume_mounted(current) or not current.parent.is_dir():
        raise ValueError("The current model storage is unavailable; reconnect it first")
    if current.exists() and any(path.name != ".decision-locks" for path in current.iterdir()):
        raise ValueError("Existing model files must be migrated before changing storage")
    metadata_path = current.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("The existing model registry could not be checked") from exc
        if metadata:
            raise ValueError("Existing registered models must be migrated before changing storage")


def _clean_completed_server_logs(directory: Path, retention_days: int | None) -> int:
    """Remove only expired, app-created log files from completed server sessions."""
    if retention_days is None:
        return 0
    cutoff = time.time() - retention_days * 24 * 60 * 60
    failures = 0
    try:
        candidates = list(directory.iterdir())
    except OSError:
        return 1
    for path in candidates:
        if not re.fullmatch(r"server-[0-9a-f]{32}\.log", path.name):
            continue
        try:
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mtime > cutoff):
                continue
            current = path.lstat()
            if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size) != (
                info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size
            ):
                continue
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            failures += 1
    return failures


class ServerController:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.host: str | None = None
        self.management_secret: str | None = None
        self.started_at: float | None = None
        self.ready_once = False
        self.log_path: Path | None = None
        self.log_cleanup_failures = 0

    def _finish_log(self) -> None:
        """Mark a log completed only after the server's process group has stopped."""
        process, path = self.process, self.log_path
        if process is None or path is None or process.poll() is None:
            return
        try:
            os.killpg(process.pid, 0)
            return
        except ProcessLookupError:
            pass
        except PermissionError:
            return
        completed = path.with_name(path.name.replace("running-", "server-", 1))
        try:
            os.link(path, completed, follow_symlinks=False)
            path.unlink()
            self.log_path = None
        except OSError:
            # An unconfirmed or unmovable log is never eligible for retention cleanup.
            pass

    def authorization(self, port: int) -> str | None:
        if self.port == port and self.process is not None and self.process.poll() is None:
            return self.management_secret
        return None

    def address(self, port: int) -> str:
        if self.port == port and self.process is not None and self.process.poll() is None:
            return self.host or "127.0.0.1"
        return _loopback_host(Settings().host)

    async def status(self) -> dict[str, Any]:
        self._finish_log()
        config = Settings()
        port = self.port if self.process and self.process.poll() is None else config.port
        managed = bool(self.process and self.process.poll() is None)
        host = self.address(port)
        try:
            health = await asyncio.to_thread(_request_json, port, "/health", timeout=2, host=host)
            if (
                not isinstance(health, dict)
                or health.get("role") != "gateway"
                or "version" not in health
            ):
                raise ValueError("Unexpected service")
            if managed:
                self.ready_once = True
            return {
                "state": "degraded" if health.get("status") != "healthy" else "connected",
                "ownership": "app" if managed else "external", "port": port,
                "url": _server_url(port, host), "health": health,
                "log_cleanup_failures": self.log_cleanup_failures,
            }
        except (HTTPException, ValueError):
            exit_code = self.process.poll() if self.process is not None else None
            if managed:
                elapsed = time.monotonic() - self.started_at if self.started_at is not None else 0
                unavailable = "degraded" if self.ready_once or elapsed >= 30 else "starting"
            else:
                unavailable = (
                    "port_conflict" if _port_is_busy(port, host)
                    else "failed" if self.process is not None and exit_code is not None
                    else "stopped"
                )
            return {
                "state": unavailable,
                "ownership": "app" if managed else None,
                "port": port, "url": _server_url(port, host), "health": None,
                "last_exit_code": exit_code if unavailable == "failed" else None,
                "log_cleanup_failures": self.log_cleanup_failures,
            }

    async def start(self) -> dict[str, Any]:
        state = await self.status()
        if state["state"] not in ("stopped", "failed"):
            raise HTTPException(status_code=409, detail="The configured port is already in use")
        config = Settings()
        if config.host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "::"):
            raise HTTPException(status_code=400, detail="Unsupported configured bind address")
        if not model_volume_mounted(config.models_dir):
            raise HTTPException(
                status_code=422, detail="Model storage volume is unavailable; reconnect it first"
            )
        log_dir = self.data_dir / "app-server-logs"
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=422, detail="Server log directory is unavailable"
            ) from exc
        if log_dir.is_symlink() or not log_dir.is_dir():
            raise HTTPException(status_code=422, detail="Server log directory is unavailable")
        self.log_cleanup_failures = _clean_completed_server_logs(
            log_dir, config.app_log_retention_days
        )
        log_path = log_dir / f"running-{secrets.token_hex(16)}.log"
        try:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise HTTPException(
                status_code=422, detail="Server log directory is unavailable"
            ) from exc
        try:
            os.fchmod(fd, 0o600)
            self.management_secret = secrets.token_urlsafe(32)
            env = public_hf_environment()
            env["MLX_SERVE_MANAGEMENT_TOKEN"] = self.management_secret
            self.process = subprocess.Popen(
                [sys.executable, "-m", "mlx_serve.app_server_runner", "--host", config.host,
                 "--port", str(config.port), "--parent", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, start_new_session=True,
                env=env,
            )
            self.log_path = log_path
            self.port = config.port
            self.host = _loopback_host(config.host)
            self.started_at = time.monotonic()
            self.ready_once = False
        finally:
            os.close(fd)
        return await self.status()

    async def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            raise HTTPException(status_code=409, detail="The app does not own a running server")
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 50)
        except subprocess.TimeoutExpired:
            # Only the process created by this agent is signalled.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.wait)
        self._finish_log()
        self.process = None
        self.port = None
        self.host = None
        self.management_secret = None
        self.started_at = None
        self.ready_once = False


class SettingsChanges(BaseModel):
    values: dict[str, Any]


class StorageMigrationRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=4096)


class SecureRequest(BaseModel):
    payload: str = Field(max_length=2 * 1024 * 1024)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    name: str | None = None
    revision: str | None = None
    keep_original: bool = False
    load_after: bool = False


class ModelCommand(BaseModel):
    name: str
    type: str


class HFCacheCleanupEntry(BaseModel):
    repo: str = Field(min_length=3, max_length=300)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    size_bytes: int = Field(ge=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelRemovalPreview(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    type: str
    size_bytes: int = Field(ge=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class TestRequest(BaseModel):
    path: str
    body: dict[str, Any]


def create_agent(token: str, data_dir: Path = DATA_DIR) -> FastAPI:
    controller = ServerController(data_dir)
    secure_channel = SecureAgentChannel(token)
    jobs: JobManager | None = None
    storage_guard = asyncio.Lock()
    migration_lock = threading.Lock()
    migration_cancel = threading.Event()
    migration_task: asyncio.Task | None = None
    migration_state: dict[str, Any] = {"status": "idle"}
    installation_guard = asyncio.Lock()
    installation: dict[str, Any] | None = None
    installation_checked_at = 0.0

    async def existing_installation() -> dict[str, Any]:
        nonlocal installation, installation_checked_at
        async with installation_guard:
            now = time.monotonic()
            if installation is None or now - installation_checked_at >= 30:
                installation = await asyncio.to_thread(_installation_snapshot)
                installation_checked_at = now
            return installation

    def migration_snapshot() -> dict[str, Any]:
        with migration_lock:
            return dict(migration_state)

    def update_migration(**changes: Any) -> None:
        with migration_lock:
            migration_state.update(changes)

    def check_not_migrating() -> None:
        if migration_snapshot()["status"] == "running":
            raise HTTPException(status_code=409, detail="Model storage migration is in progress")

    async def check_migration_ready() -> Path:
        assert jobs is not None
        state = await controller.status()
        if state["ownership"] == "external":
            raise HTTPException(status_code=409, detail="External server settings are read-only")
        if state["state"] not in ("stopped", "failed"):
            raise HTTPException(
                status_code=409, detail="Stop the server before changing model storage"
            )
        if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
            raise HTTPException(
                status_code=409, detail="Finish or cancel model preparation before changing storage"
            )
        field = settings_snapshot()["fields"]["models_dir"]
        if field["source"] in ("environment", "dotenv"):
            raise HTTPException(
                status_code=409, detail="Model storage is overridden by the environment"
            )
        return Settings().models_dir

    async def run_migration(source: Path, destination: Path) -> None:
        def progress(phase: str, copied: int, total: int) -> None:
            update_migration(phase=phase, copied_bytes=copied, size_bytes=total)

        try:
            result = await asyncio.to_thread(
                migrate_storage, source, destination, cancelled=migration_cancel, update=progress,
            )
            update_migration(status="succeeded", phase="complete", **result)
        except (OSError, ValueError) as exc:
            update_migration(status="failed", error=str(exc))
        except Exception:
            update_migration(status="failed", error="Model storage copy failed")

    async def load_prepared_model(name: str, model_type: str) -> None:
        state = await controller.status()
        if state["state"] not in ("connected", "degraded"):
            raise RuntimeError("Server is unavailable")
        port = int(state["port"])
        await _post_local_management(
            port, "/management/models/load", {"name": name, "type": model_type},
            authorization=controller.authorization(port), host=controller.address(port),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal jobs
        initialize_app_config()
        jobs = JobManager(data_dir / "jobs", load_model=load_prepared_model)
        await jobs.recover()
        yield
        if migration_task is not None and not migration_task.done():
            migration_cancel.set()
            await migration_task
        await jobs.close()
        if controller.process is not None and controller.process.poll() is None:
            await controller.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Pydantic's default errors include the submitted input, including HF tokens.
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request. Check the entered values."}
        )

    @app.middleware("http")
    async def authorize(request: Request, call_next):
        bearer = request.headers.get("authorization", "")
        if not is_local_management_request(request):
            return JSONResponse(status_code=403, content={"detail": "Local authorization required"})
        if request.url.path == "/secure":
            return await call_next(request)
        if not secrets.compare_digest(bearer, f"Bearer {token}"):
            return JSONResponse(status_code=403, content={"detail": "Local authorization required"})
        return await call_next(request)

    @app.post("/secure")
    async def secure_request(payload: SecureRequest) -> dict[str, str]:
        try:
            nonce, envelope = secure_channel.open(payload.payload)
            method, path = envelope["method"], envelope["path"]
            body = envelope.get("body")
            if (
                method not in ("GET", "POST", "PUT") or not isinstance(path, str)
                or not path.startswith("/") or path.startswith("//") or path == "/secure"
                or "?" in path or "#" in path or "\\" in path
                or (body is not None and not isinstance(body, dict))
                or (method == "GET" and body is not None)
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=403, detail="Secure management request is invalid"
            ) from exc
        transport = httpx.ASGITransport(
            app=app, client=("127.0.0.1", 1), raise_app_exceptions=False
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.request(
                method, path, json=body, headers={"Authorization": f"Bearer {token}"},
            )
        try:
            result = response.json()
        except ValueError:
            result = {"detail": "Management request failed"}
        return {"payload": secure_channel.seal(nonce, response.status_code, result)}

    async def live_port() -> int:
        state = await controller.status()
        if state["state"] not in ("connected", "degraded"):
            raise HTTPException(status_code=503, detail="Start or connect to the server first")
        return int(state["port"])

    @app.get("/status")
    async def status() -> dict[str, Any]:
        state = await controller.status()
        state["next_bind_host"] = Settings().host
        state["installation"] = await existing_installation()
        return state

    @app.post("/server/start")
    async def start_server() -> dict[str, Any]:
        async with storage_guard:
            check_not_migrating()
            return await controller.start()

    @app.post("/server/stop")
    async def stop_server() -> dict[str, str]:
        assert jobs is not None
        if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
            raise HTTPException(status_code=409, detail="Cancel or finish active jobs first")
        await controller.stop()
        return {"status": "stopped"}

    @app.get("/settings")
    async def get_settings() -> dict[str, Any]:
        try:
            port = await live_port()
            return await asyncio.to_thread(
                _request_json, port, "/management/settings",
                authorization=controller.authorization(port), host=controller.address(port),
            )
        except HTTPException:
            snapshot = settings_snapshot()
            snapshot["server_settings_available"] = False
            return snapshot

    @app.get("/runtime")
    async def runtime() -> dict[str, Any]:
        versions = {}
        for name in ("mlx-serve", "mlx", "mlx-vlm", "huggingface-hub", "transformers"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {
            "python": sys.version.split()[0], "executable": sys.executable, "packages": versions,
        }

    @app.put("/settings")
    async def put_settings(payload: SettingsChanges) -> dict[str, Any]:
        async with storage_guard:
            check_not_migrating()
            state = await controller.status()
            if state["ownership"] == "external":
                raise HTTPException(
                    status_code=409, detail="External server settings are read-only"
                )
            try:
                if "models_dir" in payload.values:
                    assert jobs is not None
                    current = Settings().models_dir
                    requested = str(payload.values["models_dir"])
                    if (
                        current.resolve() != Path(requested).expanduser().resolve()
                        and state["state"] not in ("stopped", "failed")
                    ):
                        raise ValueError("Stop the server before changing model storage")
                    _check_storage_change(
                        current, requested,
                        any(job["status"] in ACTIVE_STATES for job in jobs.list()),
                    )
                save_settings(payload.values)
                return await get_settings()
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/storage/migration")
    async def storage_migration_status() -> dict[str, Any]:
        return migration_snapshot()

    @app.post("/storage/migration/preview")
    async def storage_migration_preview(payload: StorageMigrationRequest) -> dict[str, object]:
        async with storage_guard:
            check_not_migrating()
            source = await check_migration_ready()
            try:
                return await asyncio.to_thread(
                    preview_storage_migration, source, Path(payload.destination),
                )
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/storage/migration")
    async def start_storage_migration(payload: StorageMigrationRequest) -> dict[str, Any]:
        nonlocal migration_task
        async with storage_guard:
            check_not_migrating()
            source = await check_migration_ready()
            migration_cancel.clear()
            with migration_lock:
                migration_state.clear()
                migration_state.update({
                    "status": "running", "phase": "checking", "source": str(source),
                    "destination": payload.destination, "copied_bytes": 0, "size_bytes": 0,
                })
            migration_task = asyncio.create_task(run_migration(source, Path(payload.destination)))
            return migration_snapshot()

    @app.get("/models")
    async def models() -> dict[str, Any]:
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, "/management/state", timeout=12,
            authorization=controller.authorization(port), host=controller.address(port),
        )

    @app.get("/hf-cache")
    async def hf_cache() -> dict[str, Any]:
        from mlx_serve.core.hf_cache_inventory import inspect_hf_cache

        try:
            return await asyncio.to_thread(inspect_hf_cache)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail="Hugging Face cache is unavailable"
            ) from exc

    async def hf_cache_cleanup_ready() -> Path:
        check_not_migrating()
        assert jobs is not None
        if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
            raise HTTPException(status_code=409, detail="Finish or cancel active jobs first")
        state = await controller.status()
        if state["state"] not in ("stopped", "failed"):
            raise HTTPException(status_code=409, detail="Stop the server before cleaning the cache")
        return Settings().models_dir

    @app.get("/hf-cache/cleanup")
    async def hf_cache_cleanup_preview() -> dict[str, Any]:
        from mlx_serve.core.hf_cache_cleanup import stale_hf_cache_revisions

        async with storage_guard:
            models_dir = await hf_cache_cleanup_ready()
            try:
                return await asyncio.to_thread(stale_hf_cache_revisions, models_dir)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/hf-cache/cleanup")
    async def hf_cache_cleanup(payload: HFCacheCleanupEntry) -> dict[str, Any]:
        from mlx_serve.core.hf_cache_cleanup import cleanup_stale_hf_cache_revision

        async with storage_guard:
            models_dir = await hf_cache_cleanup_ready()
            try:
                return await asyncio.to_thread(
                    cleanup_stale_hf_cache_revision, models_dir, payload.model_dump()
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def offline_models_ready() -> Path:
        check_not_migrating()
        assert jobs is not None
        if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
            raise HTTPException(status_code=409, detail="Finish or cancel active jobs first")
        state = await controller.status()
        if state["state"] not in ("stopped", "failed"):
            raise HTTPException(status_code=409, detail="Stop the server before removing models")
        return Settings().models_dir

    @app.get("/stored-models")
    async def stored_models() -> dict[str, Any]:
        from mlx_serve.core.offline_models import offline_model_list

        async with storage_guard:
            models_dir = await offline_models_ready()
            try:
                return await asyncio.to_thread(offline_model_list, models_dir)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/stored-models/preview")
    async def stored_model_preview(payload: ModelCommand) -> dict[str, Any]:
        from mlx_serve.core.offline_models import preview_model_removal

        async with storage_guard:
            models_dir = await offline_models_ready()
            try:
                return await asyncio.to_thread(
                    preview_model_removal, models_dir, payload.name, payload.type
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/stored-models/remove")
    async def stored_model_remove(payload: ModelRemovalPreview) -> dict[str, Any]:
        from mlx_serve.core.offline_models import remove_offline_model

        async with storage_guard:
            models_dir = await offline_models_ready()
            try:
                return await asyncio.to_thread(
                    remove_offline_model, models_dir, payload.model_dump()
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/originals")
    async def originals() -> dict[str, Any]:
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, "/management/originals", timeout=120,
            authorization=controller.authorization(port), host=controller.address(port),
        )

    @app.post("/originals/cleanup")
    async def cleanup_originals(payload: dict[str, Any]) -> dict[str, Any]:
        assert jobs is not None
        if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
            raise HTTPException(status_code=409, detail="Finish or cancel active jobs first")
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, "/management/originals/cleanup", payload, timeout=190,
            authorization=controller.authorization(port), host=controller.address(port),
        )

    @app.post("/models/{operation}")
    async def model_operation(operation: str, payload: ModelCommand) -> dict[str, Any]:
        if operation not in ("load", "unload", "delete", "set-default", "clear-default"):
            raise HTTPException(status_code=404, detail="Unsupported operation")
        if operation == "delete":
            assert jobs is not None
            if any(job["status"] in ACTIVE_STATES for job in jobs.list()):
                raise HTTPException(status_code=409, detail="Finish or cancel active jobs first")
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, f"/management/models/{operation}",
            payload.model_dump(), timeout=190, authorization=controller.authorization(port),
            host=controller.address(port),
        )

    @app.get("/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        assert jobs is not None
        return jobs.list()

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        assert jobs is not None
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/jobs")
    async def start_job(payload: JobRequest) -> dict[str, Any]:
        async with storage_guard:
            check_not_migrating()
            assert jobs is not None
            port = await live_port()
            server_settings = await asyncio.to_thread(
                _request_json, port, "/management/settings",
                authorization=controller.authorization(port), host=controller.address(port),
            )
            models_dir = server_settings["fields"]["models_dir"]["applied"]
            if not model_storage_available(Path(models_dir)):
                raise HTTPException(status_code=503, detail="Model storage is unavailable")
            try:
                params = payload.model_dump()
                params["models_dir"] = models_dir
                return jobs.start(params)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        assert jobs is not None
        try:
            return await jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/test")
    async def test_model(payload: TestRequest) -> dict[str, Any]:
        if payload.path not in ("/v1/decisions", "/v1/embeddings", "/v1/rerank"):
            raise HTTPException(status_code=400, detail="Unsupported test")
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, payload.path, payload.body, timeout=180,
            host=controller.address(port),
        )

    @app.get("/diagnostics")
    async def diagnostics() -> dict[str, Any]:
        assert jobs is not None
        state = await controller.status()
        return {
            "server": state, "jobs": jobs.list(),
            "note": "Request bodies, credentials and server logs are excluded.",
        }

    return app


async def _serve(token: str, parent_pid: int) -> None:
    app = create_agent(token)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.set_inheritable(False)
    config = uvicorn.Config(app, log_level="warning", access_log=False, proxy_headers=False)
    server = uvicorn.Server(config)

    async def monitor_parent() -> None:
        while not server.should_exit:
            if os.getppid() != parent_pid:
                server.should_exit = True
                return
            await asyncio.sleep(1)

    watcher = asyncio.create_task(monitor_parent())
    print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
    try:
        await server.serve(sockets=[listener])
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        listener.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if len(token) < 32:
        raise SystemExit("Missing local authorization token")
    asyncio.run(_serve(token, args.parent))


if __name__ == "__main__":
    main()
