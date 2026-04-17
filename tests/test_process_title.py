"""Tests for process title helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from mlx_serve.core.process_title import apply_process_title, build_process_title


def test_build_process_title_for_gateway():
    """Gateway processes should keep the base mlx-serve title."""
    assert build_process_title(server_role="gateway") == "mlx-serve"


def test_build_process_title_for_workers():
    """Worker processes should include their retrieval kind."""
    assert (
        build_process_title(server_role="worker", worker_kind="embedding")
        == "mlx-serve:embedding"
    )
    assert (
        build_process_title(server_role="worker", worker_kind="reranker")
        == "mlx-serve:reranker"
    )


def test_apply_process_title_updates_title_when_dependency_is_available():
    """Call setproctitle when the dependency is importable."""
    calls: list[str] = []
    fake_module = SimpleNamespace(setproctitle=calls.append)

    with (
        patch(
            "mlx_serve.core.process_title.build_process_title",
            return_value="mlx-serve:embedding",
        ),
        patch("mlx_serve.core.process_title.importlib.import_module", return_value=fake_module),
    ):
        assert apply_process_title() == "mlx-serve:embedding"

    assert calls == ["mlx-serve:embedding"]


def test_apply_process_title_is_noop_when_dependency_is_missing():
    """Keep startup working even before dependencies are refreshed."""
    with (
        patch("mlx_serve.core.process_title.build_process_title", return_value="mlx-serve"),
        patch(
            "mlx_serve.core.process_title.importlib.import_module",
            side_effect=ImportError,
        ),
    ):
        assert apply_process_title() == "mlx-serve"
