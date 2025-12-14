"""Tests for PID manager."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_serve.core.pid_manager import PIDManager, get_all_instances


class TestPIDManager:
    """Test PIDManager class."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for PID files."""
        with patch.object(PIDManager, "_pid_dir", tmp_path):
            yield tmp_path

    def test_write_and_read_pid(self, temp_dir):
        """Test writing and reading PID file."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            manager.write_pid(12345)
            assert manager.read_pid() == 12345

    def test_read_pid_not_exists(self, temp_dir):
        """Test reading non-existent PID file."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            assert manager.read_pid() is None

    def test_remove_pid(self, temp_dir):
        """Test removing PID file."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            manager.write_pid(12345)
            assert manager.remove_pid() is True
            assert manager.read_pid() is None
            assert manager.remove_pid() is False

    def test_is_process_running_current_process(self, temp_dir):
        """Test checking if current process is running."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            current_pid = os.getpid()
            manager.write_pid(current_pid)
            assert manager.is_process_running() is True

    def test_is_process_running_dead_process(self, temp_dir):
        """Test checking if dead process is running."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            # Use an extremely high PID that's unlikely to exist
            manager.write_pid(999999999)
            assert manager.is_process_running() is False

    def test_is_stale(self, temp_dir):
        """Test stale PID detection."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            # No PID file
            assert manager.is_stale() is False

            # Valid PID
            manager.write_pid(os.getpid())
            assert manager.is_stale() is False

            # Stale PID
            manager.write_pid(999999999)
            assert manager.is_stale() is True

    def test_cleanup_stale(self, temp_dir):
        """Test stale PID cleanup."""
        with patch.object(PIDManager, "_pid_dir", temp_dir):
            manager = PIDManager(8000)
            manager._pid_dir = temp_dir
            manager._pid_file = temp_dir / "mlx-serve-8000.pid"

            # Write stale PID
            manager.write_pid(999999999)
            assert manager._pid_file.exists()

            # Cleanup should remove it
            assert manager.cleanup_stale() is True
            assert not manager._pid_file.exists()

            # No more stale files
            assert manager.cleanup_stale() is False

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
