"""Tests for system memory command resolution."""

from pathlib import Path
from unittest.mock import patch

from mlx_serve.core.system_guard import _resolve_command_path


def test_resolve_command_path_prefers_path_lookup():
    """Use shutil.which results when PATH already resolves the command."""
    with patch("mlx_serve.core.system_guard.shutil.which", return_value="/usr/bin/vm_stat"):
        assert _resolve_command_path("vm_stat") == "/usr/bin/vm_stat"


def test_resolve_command_path_falls_back_to_common_system_dirs():
    """Resolve macOS tools when launchd PATH omits sbin directories."""
    with (
        patch("mlx_serve.core.system_guard.shutil.which", return_value=None),
        patch.object(Path, "exists", autospec=True) as exists,
        patch("mlx_serve.core.system_guard.os.access") as access,
    ):
        def fake_exists(path_obj: Path) -> bool:
            return str(path_obj) == "/usr/sbin/sysctl"

        exists.side_effect = fake_exists
        access.side_effect = lambda candidate, mode: str(candidate) == "/usr/sbin/sysctl"

        assert _resolve_command_path("sysctl") == "/usr/sbin/sysctl"
