"""Helpers for routing retrieval models to the correct worker type."""

from __future__ import annotations

from mlx_serve.core.model_manager import model_manager, resolve_model_alias
from mlx_serve.core.runtime_topology import RetrievalWorkerKind


def resolve_retrieval_model_type(model_name: str) -> RetrievalWorkerKind | None:
    """Resolve a model name to its retrieval worker kind."""
    resolved_name, _, resolved_type = resolve_model_alias(model_name)
    if resolved_type in ("embedding", "reranker"):
        return resolved_type

    installed_type = model_manager.get_model_type(model_name)
    if installed_type in ("embedding", "reranker"):
        return installed_type

    if resolved_name != model_name:
        installed_type = model_manager.get_model_type(resolved_name)
        if installed_type in ("embedding", "reranker"):
            return installed_type

    return None
