"""PID file management for mlx-serve server instances."""

import os
import signal
from pathlib import Path


class PIDManager:
    """Manage PID files for mlx-serve server instances."""

    def __init__(self, port: int):
        """Initialize PID manager for a specific port.

        Args:
            port: The port number the server is running on.
        """
        self.port = port
        self._pid_dir = Path.home() / ".mlx-serve"
        self._pid_file = self._pid_dir / f"mlx-serve-{port}.pid"

    @property
    def pid_file(self) -> Path:
        """Get the PID file path."""
        return self._pid_file

    def write_pid(self, pid: int | None = None) -> None:
        """Write the current process PID to file.

        Args:
            pid: Process ID to write. Defaults to current process PID.
        """
        self._pid_dir.mkdir(parents=True, exist_ok=True)
        pid = pid if pid is not None else os.getpid()
        self._pid_file.write_text(str(pid))

    def read_pid(self) -> int | None:
        """Read PID from file.

        Returns:
            The PID if file exists and is valid, None otherwise.
        """
        if not self._pid_file.exists():
            return None

        try:
            return int(self._pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def remove_pid(self) -> bool:
        """Remove the PID file.

        Returns:
            True if file was removed, False otherwise.
        """
        if self._pid_file.exists():
            self._pid_file.unlink()
            return True
        return False

    def is_process_running(self, pid: int | None = None) -> bool:
        """Check if a process with the given PID is running.

        Args:
            pid: Process ID to check. If None, reads from PID file.

        Returns:
            True if process is running, False otherwise.
        """
        if pid is None:
            pid = self.read_pid()

        if pid is None:
            return False

        try:
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def is_stale(self) -> bool:
        """Check if the PID file is stale (process not running).

        Returns:
            True if PID file exists but process is not running.
        """
        pid = self.read_pid()
        if pid is None:
            return False
        return not self.is_process_running(pid)

    def cleanup_stale(self) -> bool:
        """Remove PID file if it's stale.

        Returns:
            True if stale PID file was cleaned up, False otherwise.
        """
        if self.is_stale():
            return self.remove_pid()
        return False

    def stop_server(self, force: bool = False, timeout: int = 30) -> bool:
        """Stop the server by sending signals.

        Args:
            force: If True, send SIGKILL immediately. Otherwise, try SIGTERM first.
            timeout: Seconds to wait for graceful shutdown before forcing.

        Returns:
            True if server was stopped, False if no server was running.
        """
        import time

        pid = self.read_pid()
        if pid is None:
            return False

        if not self.is_process_running(pid):
            self.remove_pid()
            return False

        try:
            if force:
                os.kill(pid, signal.SIGKILL)
            else:
                # Try graceful shutdown first
                os.kill(pid, signal.SIGTERM)

                # Wait for process to terminate
                for _ in range(timeout * 10):  # Check every 100ms
                    time.sleep(0.1)
                    if not self.is_process_running(pid):
                        break
                else:
                    # Timeout - force kill
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)

            self.remove_pid()
            return True
        except (OSError, ProcessLookupError):
            self.remove_pid()
            return False


def get_all_instances() -> list[tuple[int, int]]:
    """Get all running mlx-serve instances.

    Returns:
        List of (port, pid) tuples for running instances.
    """
    pid_dir = Path.home() / ".mlx-serve"
    if not pid_dir.exists():
        return []

    instances = []
    for pid_file in pid_dir.glob("mlx-serve-*.pid"):
        try:
            port = int(pid_file.stem.split("-")[-1])
            pid_manager = PIDManager(port)
            pid = pid_manager.read_pid()
            if pid and pid_manager.is_process_running(pid):
                instances.append((port, pid))
            else:
                # Cleanup stale PID file
                pid_manager.remove_pid()
        except (ValueError, OSError):
            continue

    return instances
