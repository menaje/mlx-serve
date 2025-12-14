"""Tests for service manager."""

import platform
from unittest.mock import MagicMock, patch

import pytest

from mlx_serve.core.service_manager import (
    LaunchdManager,
    SystemdManager,
    get_platform_name,
    get_service_manager,
)


class TestGetServiceManager:
    """Test get_service_manager function."""

    def test_darwin(self):
        """Test macOS detection."""
        with patch("platform.system", return_value="Darwin"):
            manager = get_service_manager()
            assert isinstance(manager, LaunchdManager)

    def test_linux(self):
        """Test Linux detection."""
        with patch("platform.system", return_value="Linux"):
            manager = get_service_manager()
            assert isinstance(manager, SystemdManager)

    def test_windows(self):
        """Test Windows (unsupported)."""
        with patch("platform.system", return_value="Windows"):
            manager = get_service_manager()
            assert manager is None


class TestGetPlatformName:
    """Test get_platform_name function."""

    def test_darwin(self):
        """Test macOS name."""
        with patch("platform.system", return_value="Darwin"):
            assert get_platform_name() == "macOS (launchd)"

    def test_linux(self):
        """Test Linux name."""
        with patch("platform.system", return_value="Linux"):
            assert get_platform_name() == "Linux (systemd)"

    def test_other(self):
        """Test other platform name."""
        with patch("platform.system", return_value="FreeBSD"):
            assert get_platform_name() == "FreeBSD"


class TestLaunchdManager:
    """Test LaunchdManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temporary paths."""
        manager = LaunchdManager()
        manager._plist_path = tmp_path / "com.mlx-serve.server.plist"
        manager._log_dir = tmp_path / "logs"
        return manager

    def test_service_name(self, manager):
        """Test service name."""
        assert manager.service_name == "com.mlx-serve.server"

    def test_is_installed_false(self, manager):
        """Test is_installed when not installed."""
        assert manager.is_installed is False

    def test_is_installed_true(self, manager):
        """Test is_installed when installed."""
        manager._plist_path.parent.mkdir(parents=True, exist_ok=True)
        manager._plist_path.write_text("<plist>test</plist>")
        assert manager.is_installed is True

    def test_install(self, manager):
        """Test install creates plist file."""
        success, message = manager.install()
        assert success is True
        assert manager._plist_path.exists()
        content = manager._plist_path.read_text()
        assert "com.mlx-serve.server" in content
        assert "RunAtLoad" in content

    def test_install_with_run_at_load(self, manager):
        """Test install with run_at_load enabled."""
        success, _ = manager.install(run_at_load=True)
        assert success is True
        content = manager._plist_path.read_text()
        assert "<true/>" in content.split("RunAtLoad")[1].split("\n")[1]

    def test_uninstall(self, manager):
        """Test uninstall removes plist file."""
        manager.install()
        assert manager._plist_path.exists()

        with patch("subprocess.run"):
            success, message = manager.uninstall()

        assert success is True
        assert not manager._plist_path.exists()

    def test_status(self, manager):
        """Test status returns correct info."""
        with patch.object(manager, "is_running", False):
            status = manager.status()
            assert status["installed"] is False
            assert status["running"] is False
            assert status["service_name"] == "com.mlx-serve.server"


class TestSystemdManager:
    """Test SystemdManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temporary paths."""
        manager = SystemdManager()
        manager._service_dir = tmp_path / "systemd" / "user"
        manager._service_path = manager._service_dir / "mlx-serve.service"
        return manager

    def test_service_name(self, manager):
        """Test service name."""
        assert manager.service_name == "mlx-serve"

    def test_is_installed_false(self, manager):
        """Test is_installed when not installed."""
        assert manager.is_installed is False

    def test_is_installed_true(self, manager):
        """Test is_installed when installed."""
        manager._service_dir.mkdir(parents=True, exist_ok=True)
        manager._service_path.write_text("[Unit]\ntest")
        assert manager.is_installed is True

    def test_install(self, manager):
        """Test install creates service file."""
        with patch("subprocess.run"):
            success, message = manager.install()

        assert success is True
        assert manager._service_path.exists()
        content = manager._service_path.read_text()
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content

    def test_uninstall(self, manager):
        """Test uninstall removes service file."""
        with patch("subprocess.run"):
            manager.install()
            assert manager._service_path.exists()

            success, message = manager.uninstall()

        assert success is True
        assert not manager._service_path.exists()

    def test_status(self, manager):
        """Test status returns correct info."""
        with patch.object(manager, "is_running", False):
            status = manager.status()
            assert status["installed"] is False
            assert status["running"] is False
            assert status["service_name"] == "mlx-serve"
