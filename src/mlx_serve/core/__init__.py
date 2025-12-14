"""Core components for mlx-serve."""

# Note: ModelManager is imported lazily to avoid circular imports
# Use: from mlx_serve.core.model_manager import ModelManager

__all__ = ["ModelManager"]


def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name == "ModelManager":
        from mlx_serve.core.model_manager import ModelManager
        return ModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
