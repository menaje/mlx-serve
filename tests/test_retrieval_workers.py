"""Tests for retrieval worker supervision."""

import signal

from mlx_serve.core.retrieval_workers import (
    RETRIEVAL_WORKER_KINDS,
    RetrievalWorkerProcess,
    RetrievalWorkerSupervisor,
)


def test_find_orphaned_retrieval_worker_pids(monkeypatch):
    """Only PPID=1 retrieval workers should be treated as stale."""
    from mlx_serve.core import retrieval_workers

    monkeypatch.setattr(
        retrieval_workers.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            " 101 1 mlx-serve:embedding\n"
            " 102 1 mlx-serve:reranker\n"
            " 103 99 mlx-serve:embedding\n"
            " 104 1 mlx-serve\n"
        ),
    )

    assert retrieval_workers._find_orphaned_retrieval_worker_pids() == {
        101: "mlx-serve:embedding",
        102: "mlx-serve:reranker",
    }


def test_cleanup_orphaned_retrieval_workers_terminates_stale_pids(monkeypatch):
    """Startup cleanup should terminate stale retrieval worker processes."""
    from mlx_serve.config import settings
    from mlx_serve.core import retrieval_workers

    monkeypatch.setattr(
        retrieval_workers,
        "_find_orphaned_retrieval_worker_pids",
        lambda: {101: "mlx-serve:embedding"},
    )
    monkeypatch.setattr(settings, "retrieval_worker_shutdown_timeout_seconds", 0.5)

    alive = {101}
    signals: list[tuple[int, object]] = []

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            alive.discard(pid)
            return
        if sig == 0 and pid in alive:
            return
        raise ProcessLookupError

    monkeypatch.setattr(retrieval_workers.os, "kill", fake_kill)
    monkeypatch.setattr(retrieval_workers.time, "sleep", lambda _: None)

    retrieval_workers._cleanup_orphaned_retrieval_workers()

    assert signals == [(101, signal.SIGTERM), (101, 0)]


def test_supervisor_start_cleans_orphans_before_spawning(monkeypatch):
    """The supervisor should clean stale workers before creating new ones."""
    from mlx_serve.core import retrieval_workers

    order: list[str] = []

    def fake_cleanup() -> None:
        order.append("cleanup")

    def fake_start_worker(self, kind):
        order.append(f"start:{kind}")
        process = type("Process", (), {"pid": 100, "poll": lambda self: None})()
        return RetrievalWorkerProcess(
            kind=kind,
            host="127.0.0.1",
            port=9000,
            process=process,
        )

    monkeypatch.setattr(retrieval_workers, "_cleanup_orphaned_retrieval_workers", fake_cleanup)
    monkeypatch.setattr(RetrievalWorkerSupervisor, "_start_worker", fake_start_worker)

    supervisor = RetrievalWorkerSupervisor()
    worker_urls = supervisor.start()

    assert order[0] == "cleanup"
    assert order[1:] == [f"start:{kind}" for kind in RETRIEVAL_WORKER_KINDS]
    assert worker_urls == {
        "embedding": "http://127.0.0.1:9000",
        "reranker": "http://127.0.0.1:9000",
    }
