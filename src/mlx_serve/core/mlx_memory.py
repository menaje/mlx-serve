"""Utilities for managing MLX runtime memory."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_mx() -> Any:
    """Import mlx.core lazily so non-MLX code paths stay lightweight."""
    import mlx.core as mx

    return mx


def clear_mlx_cache(
    *,
    log: logging.Logger | None = None,
    reason: str | None = None,
) -> bool:
    """Clear free MLX cache memory if MLX is available."""
    try:
        mx = _load_mx()
    except ImportError:
        return False

    target_logger = log or logger

    try:
        cache_bytes = int(mx.get_cache_memory())
    except Exception:
        cache_bytes = None

    try:
        mx.clear_cache()
    except Exception as exc:
        suffix = f" ({reason})" if reason else ""
        target_logger.warning("Failed to clear MLX cache%s: %s", suffix, exc)
        return False

    if cache_bytes and target_logger.isEnabledFor(logging.DEBUG):
        suffix = f" ({reason})" if reason else ""
        target_logger.debug("Cleared MLX cache%s: %s bytes", suffix, cache_bytes)

    return True


def get_mlx_memory_snapshot() -> dict[str, int | bool | None]:
    """Return MLX runtime memory counters when MLX is available."""
    try:
        mx = _load_mx()
    except ImportError:
        return {"available": False}

    snapshot: dict[str, int | bool | None] = {"available": True}
    getters = {
        "active_bytes": mx.get_active_memory,
        "cache_bytes": mx.get_cache_memory,
        "peak_bytes": mx.get_peak_memory,
    }

    for key, getter in getters.items():
        try:
            snapshot[key] = int(getter())
        except Exception:
            snapshot[key] = None

    return snapshot
