"""Runtime topology helpers for gateway and worker processes."""

from __future__ import annotations

import os
from typing import Literal

from mlx_serve.config import settings

ServerRole = Literal["gateway", "worker"]
RetrievalWorkerKind = Literal["embedding", "reranker"]

SERVER_ROLE_ENV = "MLX_SERVE_SERVER_ROLE"
RETRIEVAL_WORKER_KIND_ENV = "MLX_SERVE_RETRIEVAL_WORKER_KIND"
RETRIEVAL_WORKER_KINDS: tuple[RetrievalWorkerKind, ...] = ("embedding", "reranker")


def get_server_role() -> ServerRole:
    """Return the current process role."""
    if os.getenv(SERVER_ROLE_ENV) == "worker":
        return "worker"
    return "gateway"


def get_retrieval_worker_kind() -> RetrievalWorkerKind | None:
    """Return the retrieval worker kind when running in worker mode."""
    kind = os.getenv(RETRIEVAL_WORKER_KIND_ENV)
    if kind in RETRIEVAL_WORKER_KINDS:
        return kind
    return None


def retrieval_worker_isolation_enabled() -> bool:
    """Return whether the gateway should isolate retrieval endpoints in subprocesses."""
    return get_server_role() == "gateway" and settings.retrieval_worker_isolation_enabled
