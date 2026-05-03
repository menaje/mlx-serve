"""Supervisor for internal LLM/VLM worker subprocesses."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from mlx_serve.config import settings
from mlx_serve.core.model_manager import model_manager, resolve_model_alias
from mlx_serve.core.runtime_topology import (
    GENERATION_WORKER_KIND_ENV,
    GENERATION_WORKER_KINDS,
    GENERATION_WORKER_MODEL_ENV,
    SERVER_ROLE_ENV,
    GenerationWorkerKind,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationWorkerProcess:
    """A running internal generation worker."""

    kind: GenerationWorkerKind
    host: str
    port: int
    process: subprocess.Popen
    model_name: str | None = None
    last_used_at: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def touch(self) -> None:
        """Mark this worker as recently used."""
        self.last_used_at = time.monotonic()

    def snapshot(self) -> dict[str, object]:
        """Return a small health/debug payload for the worker."""
        return {
            "url": self.base_url,
            "pid": self.process.pid,
            "alive": self.process.poll() is None,
            "model": self.model_name,
            "last_used_at": self.last_used_at,
        }


class GenerationWorkerSupervisor:
    """Start and stop dedicated LLM/VLM worker subprocesses."""

    def __init__(self) -> None:
        self._workers: dict[str, GenerationWorkerProcess] = {}
        self._lock = threading.RLock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    def start(self) -> dict[str, str]:
        """Start configured generation workers and return their base URLs."""
        _cleanup_orphaned_generation_workers()
        if settings.generation_worker_mode == "model":
            self._start_idle_reaper()
            return {}

        try:
            for kind in GENERATION_WORKER_KINDS:
                worker = self._start_worker(kind)
                with self._lock:
                    self._workers[kind] = worker
        except Exception:
            self.stop()
            raise

        with self._lock:
            return {kind: worker.base_url for kind, worker in self._workers.items()}

    def ensure_worker(self, kind: GenerationWorkerKind, model_name: str) -> str:
        """Return a worker URL, starting a model-scoped worker on demand if configured."""
        if settings.generation_worker_mode != "model":
            with self._lock:
                worker = self._workers.get(kind)
            if worker is None or worker.process.poll() is not None:
                worker = self._start_worker(kind)
                with self._lock:
                    self._workers[kind] = worker
            worker.touch()
            return worker.base_url

        self.reap_idle_workers()
        resolved_name, _, _ = resolve_model_alias(model_name)
        key = self._worker_key(kind, resolved_name)
        with self._lock:
            worker = self._workers.get(key)
        if worker is not None and worker.process.poll() is None:
            worker.touch()
            return worker.base_url

        worker = self._start_worker(kind, model_name=resolved_name)
        with self._lock:
            self._workers[key] = worker
        return worker.base_url

    def stop(self) -> None:
        """Terminate all managed workers."""
        self._reaper_stop.set()
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            self._reaper_thread.join(timeout=1.0)
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        if not workers:
            return
        self._terminate_workers(workers)

    def reap_idle_workers(self, now: float | None = None) -> list[str]:
        """Terminate idle model-scoped workers and return their worker keys."""
        if settings.generation_worker_mode != "model":
            return []
        idle_timeout = float(settings.generation_worker_idle_timeout_seconds)
        if idle_timeout <= 0:
            return []

        now = time.monotonic() if now is None else now
        expired: list[tuple[str, GenerationWorkerProcess]] = []
        with self._lock:
            for key, worker in list(self._workers.items()):
                if worker.model_name is None:
                    continue
                if worker.process.poll() is not None:
                    expired.append((key, worker))
                    self._workers.pop(key, None)
                    continue
                if worker.last_used_at and now - worker.last_used_at >= idle_timeout:
                    expired.append((key, worker))
                    self._workers.pop(key, None)

        if not expired:
            return []

        keys = [key for key, _worker in expired]
        logger.info("Reaping idle generation workers: %s", ", ".join(keys))
        self._terminate_workers([worker for _key, worker in expired])
        return keys

    def _terminate_workers(self, workers: list[GenerationWorkerProcess]) -> None:
        deadline = time.monotonic() + settings.generation_worker_shutdown_timeout_seconds
        for worker in workers:
            if worker.process.poll() is None:
                worker.process.terminate()

        for worker in workers:
            if worker.process.poll() is not None:
                continue
            timeout = max(0.0, deadline - time.monotonic())
            try:
                worker.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=1.0)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a health/debug snapshot for managed workers."""
        self.reap_idle_workers()
        with self._lock:
            return {key: worker.snapshot() for key, worker in self._workers.items()}

    def _start_idle_reaper(self) -> None:
        if settings.generation_worker_idle_timeout_seconds <= 0:
            return
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._idle_reaper_loop,
            name="generation-worker-idle-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def _idle_reaper_loop(self) -> None:
        while not self._reaper_stop.is_set():
            idle_timeout = float(settings.generation_worker_idle_timeout_seconds)
            interval = 30.0 if idle_timeout <= 0 else min(30.0, max(1.0, idle_timeout / 2))
            if self._reaper_stop.wait(interval):
                break
            self.reap_idle_workers()

    def _start_worker(
        self,
        kind: GenerationWorkerKind,
        model_name: str | None = None,
    ) -> GenerationWorkerProcess:
        host = settings.generation_worker_host
        port = _find_free_port(host)
        env = os.environ.copy()
        env[SERVER_ROLE_ENV] = "worker"
        env[GENERATION_WORKER_KIND_ENV] = kind
        env.pop("MLX_SERVE_RETRIEVAL_WORKER_KIND", None)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        preload_models = [model_name] if model_name else _select_preload_models(kind)
        if model_name:
            env[GENERATION_WORKER_MODEL_ENV] = model_name
        else:
            env.pop(GENERATION_WORKER_MODEL_ENV, None)
        if preload_models:
            env["MLX_SERVE_PRELOAD_MODELS"] = ",".join(preload_models)
        else:
            env.pop("MLX_SERVE_PRELOAD_MODELS", None)

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "mlx_serve.server:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            env=env,
        )

        worker = GenerationWorkerProcess(
            kind=kind,
            host=host,
            port=port,
            process=process,
            model_name=model_name,
        )
        worker.touch()
        _wait_for_worker_ready(worker)
        logger.info(
            "Started %s generation worker on %s (PID: %s)",
            f"{kind}:{model_name}" if model_name else kind,
            worker.base_url,
            process.pid,
        )
        return worker

    def _worker_key(self, kind: GenerationWorkerKind, model_name: str) -> str:
        return f"{kind}:{model_name}"


def _select_preload_models(kind: GenerationWorkerKind) -> list[str]:
    """Return configured preload models that belong to a generation worker."""
    selected: list[str] = []
    for model_name in settings.preload_models:
        _, _, resolved_type = resolve_model_alias(model_name)
        model_type = resolved_type or model_manager.get_model_type(model_name)
        if model_type == kind:
            selected.append(model_name)
    return selected


def _find_orphaned_generation_worker_pids() -> dict[int, str]:
    """Return orphaned generation worker processes from previous runs."""
    expected_titles = {f"mlx-serve:{kind}" for kind in GENERATION_WORKER_KINDS}

    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,command="],
            text=True,
        )
    except Exception as exc:
        logger.warning("Failed to inspect running generation workers: %s", exc)
        return {}

    stale: dict[int, str] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue

        pid_str, ppid_str, command = parts
        title = command.strip()
        if title not in expected_titles or ppid_str != "1":
            continue

        try:
            stale[int(pid_str)] = title
        except ValueError:
            continue

    return stale


def _process_exists(pid: int) -> bool:
    """Check whether a process still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _cleanup_orphaned_generation_workers() -> None:
    """Terminate orphaned generation workers left behind by prior gateway runs."""
    stale_workers = _find_orphaned_generation_worker_pids()
    if not stale_workers:
        return

    stale_pids = sorted(stale_workers)
    logger.warning(
        "Cleaning up orphaned generation workers from previous runs: %s",
        ", ".join(f"{pid}:{stale_workers[pid]}" for pid in stale_pids),
    )

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.monotonic() + settings.generation_worker_shutdown_timeout_seconds
    alive = {pid for pid in stale_pids if _process_exists(pid)}
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = {pid for pid in alive if _process_exists(pid)}

    for pid in sorted(alive):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _find_free_port(host: str) -> int:
    """Reserve a currently free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_worker_ready(worker: GenerationWorkerProcess) -> None:
    """Block until a worker reports healthy or exits."""
    deadline = time.monotonic() + settings.generation_worker_ready_timeout_seconds
    health_url = f"{worker.base_url}/health"

    while time.monotonic() < deadline:
        if worker.process.poll() is not None:
            raise RuntimeError(
                f"{worker.kind} generation worker exited during startup "
                f"(exit_code={worker.process.returncode})"
            )

        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(0.1)

    worker.process.terminate()
    worker.process.wait(timeout=1.0)
    raise RuntimeError(f"Timed out waiting for {worker.kind} generation worker at {health_url}")
