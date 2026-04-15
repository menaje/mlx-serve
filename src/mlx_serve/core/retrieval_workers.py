"""Supervisor for internal retrieval worker subprocesses."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from mlx_serve.config import settings
from mlx_serve.core.model_manager import model_manager, resolve_model_alias
from mlx_serve.core.runtime_topology import (
    RETRIEVAL_WORKER_KIND_ENV,
    RETRIEVAL_WORKER_KINDS,
    SERVER_ROLE_ENV,
    RetrievalWorkerKind,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalWorkerProcess:
    """A running internal retrieval worker."""

    kind: RetrievalWorkerKind
    host: str
    port: int
    process: subprocess.Popen

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def snapshot(self) -> dict[str, object]:
        """Return a small health/debug payload for the worker."""
        return {
            "url": self.base_url,
            "pid": self.process.pid,
            "alive": self.process.poll() is None,
        }


class RetrievalWorkerSupervisor:
    """Start and stop dedicated retrieval worker subprocesses."""

    def __init__(self) -> None:
        self._workers: dict[RetrievalWorkerKind, RetrievalWorkerProcess] = {}

    def start(self) -> dict[RetrievalWorkerKind, str]:
        """Start all retrieval workers and return their base URLs."""
        try:
            for kind in RETRIEVAL_WORKER_KINDS:
                worker = self._start_worker(kind)
                self._workers[kind] = worker
        except Exception:
            self.stop()
            raise

        return {kind: worker.base_url for kind, worker in self._workers.items()}

    def stop(self) -> None:
        """Terminate all managed workers."""
        workers = list(self._workers.values())
        if not workers:
            return

        for worker in workers:
            if worker.process.poll() is None:
                worker.process.terminate()

        deadline = time.monotonic() + settings.retrieval_worker_shutdown_timeout_seconds
        for worker in workers:
            if worker.process.poll() is not None:
                continue
            timeout = max(0.0, deadline - time.monotonic())
            try:
                worker.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=1.0)

        self._workers.clear()

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a health/debug snapshot for managed workers."""
        return {kind: worker.snapshot() for kind, worker in self._workers.items()}

    def _start_worker(self, kind: RetrievalWorkerKind) -> RetrievalWorkerProcess:
        host = settings.retrieval_worker_host
        port = _find_free_port(host)
        env = os.environ.copy()
        env[SERVER_ROLE_ENV] = "worker"
        env[RETRIEVAL_WORKER_KIND_ENV] = kind
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        preload_models = _select_preload_models(kind)
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

        worker = RetrievalWorkerProcess(
            kind=kind,
            host=host,
            port=port,
            process=process,
        )
        _wait_for_worker_ready(worker)
        logger.info(
            "Started %s retrieval worker on %s (PID: %s)",
            kind,
            worker.base_url,
            process.pid,
        )
        return worker


def _select_preload_models(kind: RetrievalWorkerKind) -> list[str]:
    """Return the configured preload models that belong to a worker kind."""
    selected: list[str] = []
    for model_name in settings.preload_models:
        _, _, resolved_type = resolve_model_alias(model_name)
        model_type = resolved_type or model_manager.get_model_type(model_name)
        if model_type == kind:
            selected.append(model_name)
    return selected


def _find_free_port(host: str) -> int:
    """Reserve a currently free local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_worker_ready(worker: RetrievalWorkerProcess) -> None:
    """Block until a worker reports healthy or exits."""
    deadline = time.monotonic() + settings.retrieval_worker_ready_timeout_seconds
    health_url = f"{worker.base_url}/health"

    while time.monotonic() < deadline:
        if worker.process.poll() is not None:
            raise RuntimeError(
                f"{worker.kind} retrieval worker exited during startup "
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
    raise RuntimeError(f"Timed out waiting for {worker.kind} retrieval worker at {health_url}")
