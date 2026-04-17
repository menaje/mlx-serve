"""Tests for service manager."""

import os
import subprocess
from unittest.mock import PropertyMock, patch

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
        assert manager._launchctl_domain == f"gui/{os.getuid()}"
        assert manager._launchctl_target == f"gui/{os.getuid()}/com.mlx-serve.server"

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
        assert "/usr/sbin" in content
        assert "/sbin" in content

    def test_install_with_run_at_load(self, manager):
        """Test install with run_at_load enabled."""
        success, _ = manager.install(run_at_load=True)
        assert success is True
        content = manager._plist_path.read_text()
        assert "<true/>" in content.split("RunAtLoad")[1].split("\n")[1]

    def test_install_uses_config_keep_alive(self, manager, monkeypatch):
        """Test install uses config keep_alive when not explicitly overridden."""
        monkeypatch.setattr("mlx_serve.core.service_manager.settings.service_keep_alive", False)
        success, _ = manager.install()
        assert success is True
        content = manager._plist_path.read_text()
        assert "<false/>" in content.split("KeepAlive")[1].split("\n")[1]

    def test_install_removes_legacy_plists(self, manager, tmp_path):
        """Test install removes legacy plist files outside the managed path."""
        legacy_path = tmp_path / "legacy.plist"
        legacy_path.write_text("<plist>legacy</plist>")
        manager._legacy_plist_paths = [legacy_path]

        with patch("subprocess.run"):
            success, message = manager.install()

        assert success is True
        assert not legacy_path.exists()
        assert "removed legacy plist" in message

    def test_uninstall(self, manager):
        """Test uninstall removes plist file."""
        manager.install()
        assert manager._plist_path.exists()

        with patch("subprocess.run"):
            success, message = manager.uninstall()

        assert success is True
        assert not manager._plist_path.exists()

    def test_start_uses_bootstrap_and_kickstart(self, manager):
        """Test launchd start uses bootstrap and kickstart."""
        manager.install()

        with patch("subprocess.run") as run, patch.object(
            manager, "_wait_for_http_ready", return_value=True
        ):
            run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="state = running\n",
                    stderr="",
                ),
            ]
            success, message = manager.start()

        assert success is True
        assert message == "Service started"
        assert run.call_args_list[0].args[0][:3] == [
            "launchctl",
            "bootstrap",
            manager._launchctl_domain,
        ]
        assert run.call_args_list[1].args[0] == [
            "launchctl",
            "kickstart",
            "-k",
            manager._launchctl_target,
        ]
        assert run.call_args_list[2].args[0] == [
            "launchctl",
            "print",
            manager._launchctl_target,
        ]

    def test_start_waits_for_running_state(self, manager):
        """Test launchd start tolerates the initial spawn-scheduled state."""
        manager.install()

        with (
            patch("subprocess.run") as run,
            patch("time.sleep"),
            patch.object(manager, "_wait_for_http_ready", return_value=True),
        ):
            run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="state = spawn scheduled\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="state = running\n",
                    stderr="",
                ),
            ]
            success, message = manager.start()

        assert success is True
        assert message == "Service started"

    def test_start_waits_for_http_ready(self, manager):
        """Test launchd start waits for the health endpoint after launchd reports running."""
        manager.install()

        with patch("subprocess.run") as run, patch.object(
            manager, "_wait_for_http_ready", return_value=True
        ) as wait_for_http_ready:
            run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="state = running\n",
                    stderr="",
                ),
            ]
            success, message = manager.start()

        assert success is True
        assert message == "Service started"
        wait_for_http_ready.assert_called_once_with()

    def test_stop_uses_bootout_target_first(self, manager):
        """Test launchd stop uses bootout with the fully-qualified target."""
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            success, message = manager.stop()

        assert success is True
        assert message == "Service stopped"
        assert run.call_args_list[0].args[0] == [
            "launchctl",
            "bootout",
            manager._launchctl_target,
        ]

    def test_is_running_checks_launchctl_print_state(self, manager):
        """Test launchd running state is read from launchctl print output."""
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="state = running\n",
                stderr="",
            )
            assert manager.is_running is True

    def test_status(self, manager):
        """Test status returns correct info."""
        with patch.object(
            type(manager), "is_running", new_callable=PropertyMock, return_value=False
        ):
            with patch.object(
                type(manager), "is_enabled", new_callable=PropertyMock, return_value=False
            ):
                with patch.object(
                    type(manager),
                    "keep_alive_enabled",
                    new_callable=PropertyMock,
                    return_value=True,
                ):
                    status = manager.status()
                    assert status["installed"] is False
                    assert status["running"] is False
                    assert status["service_name"] == "com.mlx-serve.server"
                    assert status["managed_path"] == str(manager._plist_path)
                    assert status["keep_alive"] is True


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
        with patch.object(
            type(manager), "is_running", new_callable=PropertyMock, return_value=False
        ):
            with patch.object(
                type(manager), "is_enabled", new_callable=PropertyMock, return_value=False
            ):
                status = manager.status()
                assert status["installed"] is False
                assert status["running"] is False
                assert status["service_name"] == "mlx-serve"
