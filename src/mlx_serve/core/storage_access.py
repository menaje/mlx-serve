"""Detect unavailable macOS volumes before treating them as empty storage."""

from __future__ import annotations

import os
from pathlib import Path


def model_volume_mounted(path: Path) -> bool:
    """Require paths under /Volumes/<name> to have an actual mounted volume."""
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    try:
        candidates = (absolute, absolute.resolve())
        for candidate in candidates:
            parts = candidate.parts
            if len(parts) >= 3 and parts[:2] == ("/", "Volumes"):
                if not (Path("/Volumes") / parts[2]).is_mount():
                    return False
        return True
    except OSError:
        return False


def model_storage_available(path: Path) -> bool:
    return (
        model_volume_mounted(path) and path.is_dir()
        and os.access(path, os.R_OK | os.X_OK)
    )
