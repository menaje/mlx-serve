"""Tests for PID manager."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_serve.core.pid_manager import PIDManager, get_all_instances


class TestPIDManager:
    """Test PIDManager class."""

    @pytest.fixture
    def pid_manager(self, tmp_path):
        """Create a PIDManager with temporary directory."""
        manager = PIDManager(8000)
        manager._pid_dir = tmp_path
        manager._pid_file = tmp_path / "mlx-serve-8000.pid"
        return manager

    def test_write_and_read_pid(self, pid_manager):
        """Test writing and reading PID file."""
        pid_manager.write_pid(12345)
        assert pid_manager.read_pid() == 12345

    def test_read_pid_not_exists(self, pid_manager):
        """Test reading non-existent PID file."""
        assert pid_manager.read_pid() is None

    def test_remove_pid(self, pid_manager):
        """Test removing PID file."""
        pid_manager.write_pid(12345)
        assert pid_manager.remove_pid() is True
        assert pid_manager.read_pid() is None
        assert pid_manager.remove_pid() is False

    def test_is_process_running_current_process(self, pid_manager):
        """Test checking if current process is running."""
        current_pid = os.getpid()
        pid_manager.write_pid(current_pid)
        assert pid_manager.is_process_running() is True

    def test_is_process_running_dead_process(self, pid_manager):
        """Test checking if dead process is running."""
        # Use an extremely high PID that's unlikely to exist
        pid_manager.write_pid(999999999)
        assert pid_manager.is_process_running() is False

    def test_is_stale(self, pid_manager):
        """Test stale PID detection."""
        # No PID file
        assert pid_manager.is_stale() is False

        # Valid PID
        pid_manager.write_pid(os.getpid())
        assert pid_manager.is_stale() is False

        # Stale PID
        pid_manager.write_pid(999999999)
        assert pid_manager.is_stale() is True

    def test_cleanup_stale(self, pid_manager):
        """Test stale PID cleanup."""
        # Write stale PID
        pid_manager.write_pid(999999999)
        assert pid_manager._pid_file.exists()

        # Cleanup should remove it
        assert pid_manager.cleanup_stale() is True
        assert not pid_manager._pid_file.exists()

        # No more stale files
        assert pid_manager.cleanup_stale() is False

    def test_pid_file_path(self):
        """Test PID file path generation."""
        manager = PIDManager(8080)
        assert "mlx-serve-8080.pid" in str(manager.pid_file)


class TestGetAllInstances:
    """Test get_all_instances function."""

    def test_no_instances(self, tmp_path):
        """Test with no running instances."""
        with patch("mlx_serve.core.pid_manager.Path.home", return_value=tmp_path):
            instances = get_all_instances()
            assert instances == []

    def test_with_running_instance(self, tmp_path):
        """Test with running instance."""
        pid_dir = tmp_path / ".mlx-serve"
        pid_dir.mkdir(parents=True)

        # Write current process PID
        (pid_dir / "mlx-serve-8000.pid").write_text(str(os.getpid()))

        with patch("mlx_serve.core.pid_manager.Path.home", return_value=tmp_path):
            instances = get_all_instances()
            assert len(instances) == 1
            assert instances[0] == (8000, os.getpid())
