"""Process title helpers for gateway and worker processes."""

from __future__ import annotations

import importlib
import logging

from mlx_serve.core.runtime_topology import (
    RetrievalWorkerKind,
    ServerRole,
    get_retrieval_worker_kind,
    get_server_role,
)

logger = logging.getLogger(__name__)


def build_process_title(
    server_role: ServerRole | None = None,
    worker_kind: RetrievalWorkerKind | None = None,
) -> str:
    """Build the desired process title for the current runtime role."""
    role = server_role or get_server_role()
    kind = worker_kind if worker_kind is not None else get_retrieval_worker_kind()

    if role == "worker" and kind:
        return f"mlx-serve:{kind}"

    return "mlx-serve"


def apply_process_title() -> str:
    """Best-effort process title update for ps/top visibility."""
    title = build_process_title()

    try:
        module = importlib.import_module("setproctitle")
    except ImportError:
        logger.debug("setproctitle is not installed; keeping default process title")
        return title

    set_title = getattr(module, "setproctitle", None)
    if not callable(set_title):
        logger.warning("setproctitle module is missing the setproctitle callable")
        return title

    try:
        set_title(title)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning("Failed to update process title to %s", title, exc_info=True)

    return title
