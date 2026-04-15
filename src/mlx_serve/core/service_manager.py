"""Cross-platform service management for mlx-serve."""

import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from mlx_serve.config import settings


class ServiceManager(ABC):
    """Abstract base class for OS-specific service managers."""

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Return the service name."""
        pass

    @property
    @abstractmethod
    def is_installed(self) -> bool:
        """Check if the service is installed."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the service is currently running."""
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the service is enabled to start at login."""
        pass

    @abstractmethod
    def install(self) -> tuple[bool, str]:
        """Install the service. Returns (success, message)."""
        pass

    @abstractmethod
    def uninstall(self) -> tuple[bool, str]:
        """Uninstall the service. Returns (success, message)."""
        pass

    @abstractmethod
    def start(self) -> tuple[bool, str]:
        """Start the service. Returns (success, message)."""
        pass

    @abstractmethod
    def stop(self) -> tuple[bool, str]:
        """Stop the service. Returns (success, message)."""
        pass

    @abstractmethod
    def enable(self) -> tuple[bool, str]:
        """Enable the service to start at login. Returns (success, message)."""
        pass

    @abstractmethod
    def disable(self) -> tuple[bool, str]:
        """Disable the service from starting at login. Returns (success, message)."""
        pass

    def apply(self) -> tuple[bool, str]:
        """Apply the current configuration to the service definition."""
        return self.install()

    def status(self) -> dict:
        """Get service status information."""
        return {
            "installed": self.is_installed,
            "running": self.is_running,
            "enabled": self.is_enabled,
            "service_name": self.service_name,
        }


class LaunchdManager(ServiceManager):
    """macOS launchd service manager."""

    PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-serve.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>mlx_serve</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <{run_at_load}/>
    <key>KeepAlive</key>
    <{keep_alive}/>
    <key>StandardOutPath</key>
    <string>{log_dir}/mlx-serve.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/mlx-serve.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"""

    def __init__(self):
        self._plist_path = Path.home() / "Library" / "LaunchAgents" / "com.mlx-serve.server.plist"
        self._log_dir = Path.home() / ".mlx-serve" / "logs"
        self._legacy_plist_paths = [
            Path("/Library/LaunchAgents/com.mlx-serve.server.plist"),
            Path("/Library/LaunchDaemons/com.mlx-serve.server.plist"),
        ]

    @property
    def service_name(self) -> str:
        return "com.mlx-serve.server"

    @property
    def is_installed(self) -> bool:
        return self._plist_path.exists()

    @property
    def is_running(self) -> bool:
        result = subprocess.run(
            ["launchctl", "list", self.service_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    @property
    def is_enabled(self) -> bool:
        if not self.is_installed:
            return False
        # Check RunAtLoad in plist
        content = self._plist_path.read_text()
        return "<key>RunAtLoad</key>\n    <true/>" in content

    @property
    def keep_alive_enabled(self) -> bool:
        """Check if KeepAlive is enabled in the plist."""
        if not self.is_installed:
            return False
        content = self._plist_path.read_text()
        return "<key>KeepAlive</key>\n    <true/>" in content

    def _remove_legacy_plists(self) -> list[Path]:
        """Remove unmanaged legacy plist files with the same label."""
        removed_paths: list[Path] = []
        for legacy_path in self._legacy_plist_paths:
            if legacy_path == self._plist_path or not legacy_path.exists():
                continue
            try:
                subprocess.run(
                    ["launchctl", "unload", str(legacy_path)],
                    capture_output=True,
                    text=True,
                )
                legacy_path.unlink()
                removed_paths.append(legacy_path)
            except PermissionError as exc:
                raise RuntimeError(
                    f"Legacy plist exists at {legacy_path} but could not be removed"
                ) from exc
        return removed_paths

    def install(
        self,
        run_at_load: bool | None = None,
        keep_alive: bool | None = None,
    ) -> tuple[bool, str]:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._plist_path.parent.mkdir(parents=True, exist_ok=True)
        run_at_load = settings.service_auto_start if run_at_load is None else run_at_load
        keep_alive = settings.service_keep_alive if keep_alive is None else keep_alive

        try:
            removed_paths = self._remove_legacy_plists()
        except RuntimeError as exc:
            return False, str(exc)

        plist_content = self.PLIST_TEMPLATE.format(
            python_path=sys.executable,
            log_dir=str(self._log_dir),
            run_at_load="true" if run_at_load else "false",
            keep_alive="true" if keep_alive else "false",
        )
        self._plist_path.write_text(plist_content)
        message = f"Service installed at {self._plist_path}"
        if removed_paths:
            removed = ", ".join(str(path) for path in removed_paths)
            message += f" (removed legacy plist: {removed})"
        return True, message

    def uninstall(self) -> tuple[bool, str]:
        self.stop()
        if self._plist_path.exists():
            self._plist_path.unlink()
            return True, "Service uninstalled"
        return True, "Service was not installed"

    def start(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed. Run 'mlx-serve service install' first"

        result = subprocess.run(
            ["launchctl", "load", str(self._plist_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service started"
        return False, f"Failed to start service: {result.stderr}"

    def stop(self) -> tuple[bool, str]:
        result = subprocess.run(
            ["launchctl", "unload", str(self._plist_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service stopped"
        return True, "Service may not be running"

    def enable(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed"
        # Reinstall with RunAtLoad=true
        return self.install(run_at_load=True, keep_alive=self.keep_alive_enabled)

    def disable(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed"
        # Reinstall with RunAtLoad=false
        return self.install(run_at_load=False, keep_alive=self.keep_alive_enabled)

    def status(self) -> dict:
        """Get service status information."""
        status = super().status()
        status["managed_path"] = str(self._plist_path)
        status["keep_alive"] = self.keep_alive_enabled
        return status


class SystemdManager(ServiceManager):
    """Linux systemd user service manager."""

    SERVICE_TEMPLATE = """[Unit]
Description=MLX-Serve Embedding Server
After=network.target

[Service]
Type=simple
ExecStart={python_path} -m mlx_serve start --foreground
Restart=on-failure
RestartSec=5
Environment=PATH={home}/.local/bin:/usr/bin

[Install]
WantedBy=default.target
"""

    def __init__(self):
        self._service_dir = Path.home() / ".config" / "systemd" / "user"
        self._service_path = self._service_dir / "mlx-serve.service"

    @property
    def service_name(self) -> str:
        return "mlx-serve"

    @property
    def is_installed(self) -> bool:
        return self._service_path.exists()

    @property
    def is_running(self) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", self.service_name],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "active"

    @property
    def is_enabled(self) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "is-enabled", self.service_name],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "enabled"

    def install(self) -> tuple[bool, str]:
        self._service_dir.mkdir(parents=True, exist_ok=True)

        service_content = self.SERVICE_TEMPLATE.format(
            python_path=sys.executable,
            home=Path.home(),
        )
        self._service_path.write_text(service_content)

        # Reload systemd
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        return True, f"Service installed at {self._service_path}"

    def uninstall(self) -> tuple[bool, str]:
        self.stop()
        self.disable()
        if self._service_path.exists():
            self._service_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            return True, "Service uninstalled"
        return True, "Service was not installed"

    def start(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed. Run 'mlx-serve service install' first"

        result = subprocess.run(
            ["systemctl", "--user", "start", self.service_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service started"
        return False, f"Failed to start service: {result.stderr}"

    def stop(self) -> tuple[bool, str]:
        result = subprocess.run(
            ["systemctl", "--user", "stop", self.service_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service stopped"
        return True, "Service may not be running"

    def enable(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed"

        result = subprocess.run(
            ["systemctl", "--user", "enable", self.service_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service enabled (will start at login)"
        return False, f"Failed to enable service: {result.stderr}"

    def disable(self) -> tuple[bool, str]:
        if not self.is_installed:
            return False, "Service not installed"

        result = subprocess.run(
            ["systemctl", "--user", "disable", self.service_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "Service disabled (won't start at login)"
        return False, f"Failed to disable service: {result.stderr}"


def get_service_manager() -> ServiceManager | None:
    """Get the appropriate service manager for the current OS.

    Returns:
        ServiceManager instance or None if unsupported OS.
    """
    system = platform.system()
    if system == "Darwin":
        return LaunchdManager()
    elif system == "Linux":
        return SystemdManager()
    return None


def get_platform_name() -> str:
    """Get human-readable platform name."""
    system = platform.system()
    if system == "Darwin":
        return "macOS (launchd)"
    elif system == "Linux":
        return "Linux (systemd)"
    return system
