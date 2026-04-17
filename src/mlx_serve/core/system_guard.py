"""System load monitoring and memory-pressure admission guard."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from mlx_serve.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemorySnapshot:
    """Best-effort process and system memory snapshot."""

    process_rss_bytes: int | None
    system_total_bytes: int | None
    system_available_bytes: int | None
    sampled_at: float

    @property
    def process_fraction(self) -> float | None:
        if not self.process_rss_bytes or not self.system_total_bytes:
            return None
        return self.process_rss_bytes / self.system_total_bytes

    @property
    def available_fraction(self) -> float | None:
        if self.system_available_bytes is None or not self.system_total_bytes:
            return None
        return self.system_available_bytes / self.system_total_bytes


def _format_bytes(value: int | None) -> str:
    """Render bytes in a compact human-friendly form."""
    if value is None:
        return "unknown"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.1f}{unit}"


def _run_command(*args: str) -> str | None:
    """Run a short system command and return stdout on success."""
    command = _resolve_command_path(args[0])
    try:
        result = subprocess.run(
            (command, *args[1:]),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    return output or None


def _resolve_command_path(command: str) -> str:
    """Resolve system utilities even when launchd PATH omits sbin directories."""
    if os.path.sep in command:
        return command

    resolved = shutil.which(command)
    if resolved:
        return resolved

    for directory in (
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ):
        candidate = Path(directory) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    return command


def _read_process_rss_bytes() -> int | None:
    """Return current process RSS in bytes."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
        except (OSError, ValueError):
            pass

    rss_output = _run_command("ps", "-o", "rss=", "-p", str(os.getpid()))
    if rss_output is None:
        return None

    try:
        return int(rss_output) * 1024
    except ValueError:
        return None


def _read_linux_memory() -> tuple[int | None, int | None]:
    """Return total and available memory on Linux."""
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None, None

    total_bytes: int | None = None
    available_bytes: int | None = None
    try:
        for line in meminfo_path.read_text().splitlines():
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not parts:
                continue
            value_bytes = int(parts[0]) * 1024
            if key == "MemTotal":
                total_bytes = value_bytes
            elif key == "MemAvailable":
                available_bytes = value_bytes
        return total_bytes, available_bytes
    except (OSError, ValueError):
        return None, None


def _read_macos_memory() -> tuple[int | None, int | None]:
    """Return total and estimated available memory on macOS."""
    total_output = _run_command("sysctl", "-n", "hw.memsize")
    vm_stat_output = _run_command("vm_stat")
    if total_output is None or vm_stat_output is None:
        return None, None

    try:
        total_bytes = int(total_output)
    except ValueError:
        return None, None

    lines = vm_stat_output.splitlines()
    if not lines:
        return total_bytes, None

    page_size_match = re.search(r"page size of (\d+) bytes", lines[0])
    if not page_size_match:
        return total_bytes, None

    page_size = int(page_size_match.group(1))
    page_counts: dict[str, int] = {}
    for line in lines[1:]:
        match = re.match(r"Pages ([^:]+):\s+(\d+)\.", line)
        if match:
            page_counts[match.group(1).strip().lower()] = int(match.group(2))

    available_pages = (
        page_counts.get("free", 0)
        + page_counts.get("inactive", 0)
        + page_counts.get("speculative", 0)
        + page_counts.get("purgeable", 0)
    )
    return total_bytes, available_pages * page_size


def collect_memory_snapshot() -> MemorySnapshot:
    """Collect a best-effort memory snapshot."""
    process_rss_bytes = _read_process_rss_bytes()
    total_bytes, available_bytes = _read_linux_memory()
    if total_bytes is None:
        total_bytes, available_bytes = _read_macos_memory()

    return MemorySnapshot(
        process_rss_bytes=process_rss_bytes,
        system_total_bytes=total_bytes,
        system_available_bytes=available_bytes,
        sampled_at=time.time(),
    )


def _build_memory_overload_reason(snapshot: MemorySnapshot) -> str | None:
    """Translate a memory snapshot into an overload reason."""
    if not settings.memory_guard_enabled:
        return None

    process_fraction = snapshot.process_fraction
    available_fraction = snapshot.available_fraction

    if (
        settings.memory_min_available_fraction is not None
        and available_fraction is not None
        and available_fraction <= settings.memory_min_available_fraction
    ):
        available_pct = available_fraction * 100
        threshold_pct = settings.memory_min_available_fraction * 100
        return (
            "memory pressure detected: "
            f"available={_format_bytes(snapshot.system_available_bytes)} "
            f"({available_pct:.1f}%), threshold={threshold_pct:.1f}%"
        )

    if (
        settings.memory_process_limit_fraction is not None
        and process_fraction is not None
        and process_fraction >= settings.memory_process_limit_fraction
    ):
        process_pct = process_fraction * 100
        threshold_pct = settings.memory_process_limit_fraction * 100
        return (
            "memory pressure detected: "
            f"process_rss={_format_bytes(snapshot.process_rss_bytes)} "
            f"({process_pct:.1f}%), threshold={threshold_pct:.1f}%"
        )

    return None


class MemoryPressureMonitor:
    """Background monitor that samples memory and exposes overload state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MemorySnapshot(
            process_rss_bytes=None,
            system_total_bytes=None,
            system_available_bytes=None,
            sampled_at=0.0,
        )
        self._reason: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background sampling."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self.sample()

    def stop(self) -> None:
        """Stop background sampling."""
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()

        if thread and thread.is_alive():
            thread.join(timeout=2)

    def reset(self) -> None:
        """Reset monitor state. Intended for tests."""
        self.stop()
        with self._lock:
            self._snapshot = MemorySnapshot(
                process_rss_bytes=None,
                system_total_bytes=None,
                system_available_bytes=None,
                sampled_at=0.0,
            )
            self._reason = None

    def _run(self) -> None:
        while not self._stop_event.wait(settings.memory_poll_interval_seconds):
            self.sample()

    def sample(self) -> MemorySnapshot:
        """Refresh and store the current snapshot."""
        snapshot = collect_memory_snapshot()
        reason = _build_memory_overload_reason(snapshot)

        with self._lock:
            previous_reason = self._reason
            self._snapshot = snapshot
            self._reason = reason

        if snapshot.process_rss_bytes is not None:
            try:
                from mlx_serve.core.metrics import update_memory_usage

                update_memory_usage(snapshot.process_rss_bytes)
            except Exception:
                logger.debug("Failed to update memory metrics", exc_info=True)

        if reason != previous_reason:
            if reason:
                logger.warning(f"Admission guard enabled: {reason}")
            elif previous_reason:
                logger.info("Admission guard cleared")

        return snapshot

    def snapshot(self) -> MemorySnapshot:
        """Return the latest snapshot, sampling once if empty."""
        with self._lock:
            snapshot = self._snapshot

        if snapshot.sampled_at == 0:
            return self.sample()
        return snapshot

    def overload_reason(self) -> str | None:
        """Return the current overload reason, if any."""
        self.snapshot()
        with self._lock:
            return self._reason

    def health_payload(self) -> dict:
        """Return monitor state for diagnostics endpoints."""
        snapshot = self.snapshot()
        with self._lock:
            reason = self._reason

        return {
            "enabled": settings.memory_guard_enabled,
            "overloaded": reason is not None,
            "reason": reason,
            "snapshot": asdict(snapshot),
        }


memory_monitor = MemoryPressureMonitor()
