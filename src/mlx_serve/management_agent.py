"""App-owned, authenticated loopback agent for server and model operations."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mlx_serve.config import Settings
from mlx_serve.core.local_access import is_local_management_request, open_local_http
from mlx_serve.core.managed_config import (
    initialize_app_config,
    save_settings,
    settings_snapshot,
)
from mlx_serve.core.managed_jobs import ACTIVE_STATES, JobManager
from mlx_serve.core.storage_access import model_storage_available, model_volume_mounted

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


class ServerController:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.host: str | None = None
        self.management_secret: str | None = None
        self.started_at: float | None = None
        self.ready_once = False

    def authorization(self, port: int) -> str | None:
        if self.port == port and self.process is not None and self.process.poll() is None:
            return self.management_secret
        return None

    def address(self, port: int) -> str:
        if self.port == port and self.process is not None and self.process.poll() is None:
            return self.host or "127.0.0.1"
        return _loopback_host(Settings().host)

    async def status(self) -> dict[str, Any]:
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.data_dir / "app-server.log"
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            self.management_secret = secrets.token_urlsafe(32)
            env = os.environ.copy()
            env["MLX_SERVE_MANAGEMENT_TOKEN"] = self.management_secret
            self.process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "mlx_serve.server:app", "--host", config.host,
                 "--port", str(config.port), "--no-access-log", "--no-proxy-headers"],
                stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, start_new_session=True,
                env=env,
            )
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
        self.process = None
        self.port = None
        self.host = None
        self.management_secret = None
        self.started_at = None
        self.ready_once = False


class SettingsChanges(BaseModel):
    values: dict[str, Any]


class JobRequest(BaseModel):
    repo: str
    name: str | None = None
    revision: str | None = None
    keep_original: bool = False
    token: str | None = Field(default=None, max_length=4096)


class ModelCommand(BaseModel):
    name: str
    type: str


class TestRequest(BaseModel):
    path: str
    body: dict[str, Any]


def create_agent(token: str, data_dir: Path = DATA_DIR) -> FastAPI:
    controller = ServerController(data_dir)
    jobs: JobManager | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal jobs
        initialize_app_config()
        jobs = JobManager(data_dir / "jobs")
        await jobs.recover()
        yield
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
        if not is_local_management_request(request) or not secrets.compare_digest(
            bearer, f"Bearer {token}"
        ):
            return JSONResponse(status_code=403, content={"detail": "Local authorization required"})
        return await call_next(request)

    async def live_port() -> int:
        state = await controller.status()
        if state["state"] not in ("connected", "degraded"):
            raise HTTPException(status_code=503, detail="Start or connect to the server first")
        return int(state["port"])

    @app.get("/status")
    async def status() -> dict[str, Any]:
        state = await controller.status()
        state["next_bind_host"] = Settings().host
        state["installation"] = {
            "config_path": str(Path.home() / ".mlx-serve" / "config.yaml"),
            "homebrew_cli": next((p for p in (
                "/opt/homebrew/bin/mlx-serve", "/usr/local/bin/mlx-serve"
            ) if Path(p).exists()), None),
            "cli_in_path": shutil.which("mlx-serve"),
        }
        return state

    @app.post("/server/start")
    async def start_server() -> dict[str, Any]:
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
        state = await controller.status()
        if state["ownership"] == "external":
            raise HTTPException(status_code=409, detail="External server settings are read-only")
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

    @app.get("/models")
    async def models() -> dict[str, Any]:
        port = await live_port()
        return await asyncio.to_thread(
            _request_json, port, "/management/state", timeout=12,
            authorization=controller.authorization(port), host=controller.address(port),
        )

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
            params = payload.model_dump(exclude={"token"})
            params["models_dir"] = models_dir
            return jobs.start(params, token=payload.token)
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
            await asyncio.sleep(1)
            if os.getppid() != parent_pid:
                server.should_exit = True

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
