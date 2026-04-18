"""Tests for MLX memory helpers."""

import logging


def test_clear_mlx_cache_returns_false_when_mlx_is_unavailable(monkeypatch):
    """Missing MLX should be treated as a no-op."""
    from mlx_serve.core import mlx_memory

    def fail_import():
        raise ImportError("mlx not installed")

    monkeypatch.setattr(mlx_memory, "_load_mx", fail_import)

    assert mlx_memory.clear_mlx_cache() is False


def test_clear_mlx_cache_invokes_runtime_clear(monkeypatch, caplog):
    """The helper should delegate to mlx.core.clear_cache()."""
    from mlx_serve.core import mlx_memory

    calls: list[str] = []

    class FakeMx:
        @staticmethod
        def get_cache_memory() -> int:
            return 4096

        @staticmethod
        def clear_cache() -> None:
            calls.append("clear_cache")

    monkeypatch.setattr(mlx_memory, "_load_mx", lambda: FakeMx)

    with caplog.at_level(logging.DEBUG):
        assert mlx_memory.clear_mlx_cache(reason="test") is True

    assert calls == ["clear_cache"]
    assert "Cleared MLX cache (test): 4096 bytes" in caplog.text
