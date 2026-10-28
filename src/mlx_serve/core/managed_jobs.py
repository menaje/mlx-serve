"""Durable, cancellable local model-preparation jobs."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from mlx_serve.core.decision_prepare_worker import D3_REPO, MLX_REPO, SOURCE_REPO
from mlx_serve.core.storage_access import model_storage_available

ACTIVE_STATES = ("queued", "running", "cancelling", "recovery_pending")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any, token: str | None = None) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, token) for item in value]
    if isinstance(value, str):
        if token:
            value = value.replace(token, "[redacted]")
        value = re.sub(r"hf_[A-Za-z0-9_]+", "[redacted]", value)
        return value[:4096]
    return value


def _error_guidance(message: str) -> tuple[str, str]:
    text = message.lower()
    if any(part in text for part in ("gated", "403", "access denied", "forbidden")):
        return "access_approval", "Check model access approval and token permissions."
    if any(part in text for part in ("401", "unauthorized", "authentication", "invalid token")):
        return "authentication", "Check or replace the Hugging Face token."
    if any(part in text for part in ("permission denied", "read-only file system")):
        return "storage_permission", "Check access to the model storage directory."
    if any(part in text for part in ("no space", "disk", "insufficient free")):
        return "disk", "Check available storage and reconnect the model volume before retrying."
    if any(part in text for part in ("memory", "out of memory")):
        return "memory", "Unload other models or free memory before retrying."
    if any(part in text for part in ("connection", "network", "offline", "name resolution")):
        return "network", "Check the network connection and retry."
    if any(part in text for part in ("missing", "not found", "404")):
        return "missing_file", "Check the repository, revision, and required model files."
    if any(part in text for part in ("timeout", "timed out")):
        return "timeout", "Check the network and runtime, then retry."
    if any(part in text for part in ("unsupported", "not supported", "architecture")):
        return "unsupported", "Choose a supported d1 or d3 decision model."
    if any(part in text for part in ("validation", "verification", "checksum", "manifest")):
        return "validation", "Check the source and revision; retry to obtain verified files."
    if any(part in text for part in ("conversion", "convert")):
        return "conversion", "Check runtime versions and available resources, then retry."
    return "preparation", "Inspect the preparation events and retry after correcting the cause."


def _process_identity(pid: int) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
        line = result.stdout.strip()
        # lstart has five whitespace-separated components.
        parts = line.split(maxsplit=5)
        if result.returncode or len(parts) != 6:
            return None
        return " ".join(parts[:5]), parts[5]
    except (OSError, subprocess.SubprocessError):
        return None


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _job_worker_pids(job_id: str) -> list[int] | None:
    """Find only converter processes whose private staging path includes this job ID."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=3, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    found = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if (
            len(parts) == 2 and parts[0].isdigit()
            and "mlx_serve.core.decision_prepare_worker" in parts[1]
            and f".decision-prepare-{job_id}-" in parts[1]
        ):
            found.append(int(parts[0]))
    return found


async def _stop_job_workers(job_id: str) -> bool:
    pids = _job_worker_pids(job_id)
    if pids is None:
        return False
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        remaining = _job_worker_pids(job_id)
        if remaining is None:
            return False
        if not remaining:
            return True
        await asyncio.sleep(0.1)
    for pid in remaining:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    for _ in range(20):
        remaining = _job_worker_pids(job_id)
        if remaining is None:
            return False
        if not remaining:
            return True
        await asyncio.sleep(0.1)
    return False


class JobManager:
    """Run preparation in isolated process groups; keep history in SQLite."""

    def __init__(
        self, data_dir: Path, *, load_model: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.data_dir = data_dir
        self._load_model = load_model
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = (data_dir / "jobs.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise RuntimeError("Another local manager is already running") from exc
        self.db_path = data_dir / "jobs.sqlite3"
        self._tasks: dict[str, asyncio.Task] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._load_tasks: dict[str, asyncio.Task] = {}
        self._recovery_lock = asyncio.Lock()
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request_key TEXT NOT NULL, params TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, pid INTEGER, process_start TEXT,
                    result TEXT, error TEXT
                );
                DROP INDEX IF EXISTS active_job_request;
                CREATE UNIQUE INDEX active_job_request ON jobs(request_key)
                    WHERE status IN ('queued', 'running', 'cancelling', 'recovery_pending');
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    time TEXT NOT NULL, payload TEXT NOT NULL
                );
                """
            )
        os.chmod(self.db_path, 0o600)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now()
        keys = ", ".join(f"{key} = ?" for key in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {keys} WHERE id = ?", (*fields.values(), job_id))

    def _event(self, job_id: str, event: dict[str, Any], token: str | None) -> None:
        sanitized = _sanitize(event, token)
        with self._db() as db:
            db.execute(
                "INSERT INTO events (job_id, time, payload) VALUES (?, ?, ?)",
                (job_id, _now(), json.dumps(sanitized, ensure_ascii=False)),
            )
        stage = sanitized.get("status")
        if stage and stage not in ("error", "success", "resources"):
            self._update(job_id, stage=str(stage))

    async def _load_prepared_model(
        self, job_id: str, result: dict[str, Any], token: str | None = None,
    ) -> dict[str, Any]:
        """Record loading separately from the outcome of verified preparation."""
        if result.get("auto_load") in ("succeeded", "failed"):
            return result
        self._event(job_id, {"status": "loading"}, token)
        try:
            name = result.get("name")
            current = self.get(job_id)
            if (
                self._load_model is None or not isinstance(name, str) or not name
                or (current is not None and current["status"] == "cancelling")
            ):
                raise ValueError("Automatic loading is unavailable")
            load_task = asyncio.create_task(self._load_model(name))
            self._load_tasks[job_id] = load_task
            try:
                await load_task
            finally:
                self._load_tasks.pop(job_id, None)
            state = "succeeded"
            event = {"status": "load_succeeded"}
        except (Exception, asyncio.CancelledError):
            state = "failed"
            event = {
                "status": "load_failed",
                "message": (
                    "Automatic loading failed. Check the server and load the model manually."
                ),
            }
        result = {**result, "auto_load": state}
        self._event(job_id, event, token)
        self._update(job_id, result=json.dumps(result, ensure_ascii=False))
        return result

    def list(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._decode(row) for row in rows]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            events = db.execute(
                "SELECT time, payload FROM events WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        result = self._decode(row)
        result["events"] = [{"time": e["time"], **json.loads(e["payload"])} for e in events]
        return result

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["params"] = json.loads(data["params"])
        if data["result"]:
            data["result"] = json.loads(data["result"])
        if data["error"]:
            data["error_category"], data["suggested_action"] = _error_guidance(data["error"])
        data.pop("request_key", None)
        data.pop("process_start", None)
        return data

    async def recover(self) -> None:
        """Stop only identifiable orphaned runners before marking jobs interrupted."""
        async with self._recovery_lock:
            await self._recover()

    async def _recover(self) -> None:
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE status IN "
                "('queued', 'running', 'cancelling', 'recovery_pending')"
            ).fetchall()
        for row in rows:
            if row["id"] in self._tasks:
                continue
            pid = row["pid"]
            identity = _process_identity(pid) if pid else None
            expected = row["process_start"]
            ours = (
                identity is not None
                and (expected is None or identity[0] == expected)
                and "mlx_serve.core.managed_job_runner" in identity[1]
                and row["id"] in identity[1]
            )
            if ours:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGTERM)
                for _ in range(50):
                    if _process_identity(pid) is None:
                        break
                    await asyncio.sleep(0.1)
                identity = _process_identity(pid)
                if (
                    identity is not None and (expected is None or identity[0] == expected)
                    and "mlx_serve.core.managed_job_runner" in identity[1]
                    and row["id"] in identity[1]
                ):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(pid, signal.SIGKILL)
                    for _ in range(20):
                        if _process_identity(pid) is None:
                            break
                        await asyncio.sleep(0.1)
            identity = _process_identity(pid) if pid else None
            runner_unknown = bool(pid and _process_exists(pid) and (
                identity is None
                or (expected is not None and identity[0] == expected)
                or (expected is None and row["id"] in identity[1])
            ))
            workers_stopped = await _stop_job_workers(row["id"])
            cleaned = (
                not runner_unknown and workers_stopped
                and self._cleanup_stage(row["id"], json.loads(row["params"]))
            )
            if cleaned:
                params = json.loads(row["params"])
                result = json.loads(row["result"]) if row["result"] else None
                if isinstance(result, dict) and params.get("load_after"):
                    result = await self._load_prepared_model(row["id"], result)
                self._update(
                    row["id"], status="succeeded" if row["result"] else "interrupted",
                    stage="complete" if row["result"] else "interrupted", pid=None,
                    result=json.dumps(result, ensure_ascii=False) if result else None,
                    error=None if row["result"] else (
                        "The manager stopped during preparation. Check the model and retry."
                    ),
                )
            else:
                self._update(
                    row["id"], status="recovery_pending", stage="recovery_pending",
                    error="Cleanup could not be confirmed. Reconnect storage and retry recovery.",
                )

    def start(self, params: dict[str, Any], token: str | None = None) -> dict[str, Any]:
        allowed = {"repo", "name", "revision", "keep_original", "models_dir", "load_after"}
        if params.keys() - allowed:
            raise ValueError("Unsupported preparation options")
        repo = params.get("repo")
        if repo not in {SOURCE_REPO, MLX_REPO, D3_REPO}:
            raise ValueError("Only validated d1 and d3 repositories are supported")
        name = params.get("name") or None
        if name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid local model name")
        revision = params.get("revision") or None
        if revision is not None and (not isinstance(revision, str) or len(revision) > 255):
            raise ValueError("Invalid revision")
        models_dir = str(Path(params["models_dir"]).expanduser())
        if not Path(models_dir).is_absolute():
            raise ValueError("Model directory must be absolute")
        normalized = {
            "repo": repo, "name": name, "revision": revision,
            "keep_original": bool(params.get("keep_original", False)) if repo == SOURCE_REPO
            else False, "models_dir": models_dir,
            "load_after": bool(params.get("load_after", False)),
        }
        # Two names still refer to the same underlying source and conversion.
        dedupe = {
            key: value for key, value in normalized.items() if key not in ("name", "load_after")
        }
        request_key = json.dumps(dedupe, sort_keys=True)
        job_id = uuid.uuid4().hex
        now = _now()
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO jobs (id, request_key, params, status, stage, created_at, "
                    "updated_at) VALUES (?, ?, ?, 'queued', 'queued', ?, ?)",
                    (job_id, request_key, json.dumps(normalized), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("This source and conversion already have an active job") from exc
        task = asyncio.create_task(self._run(job_id, normalized, token))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        result = self.get(job_id)
        assert result is not None
        return result

    async def _run(self, job_id: str, params: dict[str, Any], token: str | None) -> None:
        process: asyncio.subprocess.Process | None = None
        result: dict[str, Any] | None = None
        error: str | None = None
        final_status = "failed"
        try:
            env = os.environ.copy()
            env.pop("MLX_SERVE_MANAGEMENT_TOKEN", None)
            env["MLX_SERVE_MODELS_DIR"] = params["models_dir"]
            if token:
                env["HF_TOKEN"] = token
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mlx_serve.core.managed_job_runner",
                "--job-id", job_id, "--parent", str(os.getpid()),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, env=env, start_new_session=True,
            )
            self._processes[job_id] = process
            identity = _process_identity(process.pid)
            self._update(
                job_id, status="running", stage="starting", pid=process.pid,
                process_start=identity[0] if identity else None,
            )
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write((json.dumps(params) + "\n").encode())
            await process.stdin.drain()
            process.stdin.close()
            while line := await process.stdout.readline():
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                event = _sanitize(event, token)
                self._event(job_id, event, token)
                if event.get("status") == "success":
                    result = event
                    self._update(job_id, result=json.dumps(result, ensure_ascii=False))
                elif event.get("status") == "error":
                    error = str(event.get("message", "Preparation failed"))
            await process.wait()
            current = self.get(job_id)
            cancelling = current is not None and current["status"] == "cancelling"
            # Publication precedes the success event.  A late cancellation must
            # not report "cancelled" after the verified model was registered.
            if result is not None:
                final_status = "succeeded"
            elif cancelling:
                final_status = "cancelled"
            else:
                error = error or "Preparation process ended unexpectedly"
        except Exception:
            error = "Could not start or monitor the preparation process"
        finally:
            if process is not None and process.returncode is None:
                await self._stop_process(process)
            current = self.get(job_id)
            if result is None and current is not None and current["status"] == "cancelling":
                final_status = "cancelled"
            workers_stopped = await _stop_job_workers(job_id)
            cleaned = workers_stopped and self._cleanup_stage(job_id, params)
            if cleaned:
                if final_status == "succeeded" and result is not None and params["load_after"]:
                    result = await self._load_prepared_model(job_id, result, token)
                self._update(
                    job_id, status=final_status,
                    stage="complete" if final_status == "succeeded" else final_status,
                    pid=None, result=json.dumps(result, ensure_ascii=False) if result else None,
                    error=error if final_status == "failed" else None,
                )
            else:
                self._update(
                    job_id, status="recovery_pending", stage="recovery_pending",
                    result=json.dumps(result, ensure_ascii=False) if result else None,
                    error="Cleanup could not be confirmed. Reconnect storage and retry recovery.",
                )
            self._processes.pop(job_id, None)

    @staticmethod
    def _cleanup_stage(job_id: str, params: dict[str, Any]) -> bool:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return False
        root = Path(params["models_dir"])
        if not model_storage_available(root):
            return False
        try:
            for stage in root.glob(f".decision-prepare-{job_id}-*"):
                if stage.is_dir() and not stage.is_symlink():
                    shutil.rmtree(stage)
            return not any(root.glob(f".decision-prepare-{job_id}-*"))
        except OSError:
            return False

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        task = self._tasks.get(job_id)
        if job["status"] == "recovery_pending" and task is None:
            await self.recover()
            result = self.get(job_id)
            assert result is not None
            return result
        if job["status"] not in ACTIVE_STATES or task is None:
            raise ValueError("Job is not running")
        self._update(job_id, status="cancelling", stage="cancelling")
        load_task = self._load_tasks.get(job_id)
        if load_task is not None:
            load_task.cancel()
        process = self._processes.get(job_id)
        if process is None:
            # Wait until the scheduled coroutine has started its process.
            while not task.done() and job_id not in self._processes:
                await asyncio.sleep(0.01)
            process = self._processes.get(job_id)
            if process is not None:
                self._update(job_id, status="cancelling", stage="cancelling")
        if process is not None:
            await self._stop_process(process)
        await task
        result = self.get(job_id)
        assert result is not None
        return result

    async def close(self) -> None:
        for job_id in list(self._tasks):
            with contextlib.suppress(KeyError, ValueError):
                await self.cancel(job_id)
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()
