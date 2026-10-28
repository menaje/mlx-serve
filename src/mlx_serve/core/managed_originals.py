"""Conservative cleanup of unreferenced retained d1 source directories."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mlx_serve.core.storage_access import model_storage_available

_ORIGINAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*-[0-9a-f]{40}$")


@contextmanager
def _cleanup_locks(models_dir: Path) -> Iterator[None]:
    if not model_storage_available(models_dir):
        raise ValueError("Model storage is unavailable")
    lock_dir = models_dir / ".decision-locks"
    if lock_dir.is_symlink():
        raise ValueError("Decision preparation lock directory is invalid")
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / "d1-preparation.lock").open("a+") as preparation:
        try:
            fcntl.flock(preparation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Finish or cancel model preparation before cleaning originals"
            ) from exc
        try:
            with (models_dir.parent / ".decision-metadata.lock").open("a+") as registry:
                fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
                try:
                    if not model_storage_available(models_dir):
                        raise ValueError("Model storage is unavailable")
                    yield
                finally:
                    fcntl.flock(registry.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(preparation.fileno(), fcntl.LOCK_UN)


def _referenced_originals(models_dir: Path) -> list[Path]:
    path = models_dir.parent / "metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Model registry is invalid")
        result = []
        for model in metadata.values():
            if not isinstance(model, dict):
                raise ValueError("Model registry is invalid")
            original = model.get("original_path")
            if original is not None:
                if not isinstance(original, str) or not Path(original).is_absolute():
                    raise ValueError("Model original references are invalid")
                result.append(Path(original).resolve())
        return result
    except (OSError, ValueError) as exc:
        raise ValueError("Model registry could not be checked") from exc


def _tree_snapshot(path: Path, parent_device: int) -> tuple[int, str] | None:
    """Reject links, special files and mounts; report logical bytes, not reclaimed bytes."""
    digest = hashlib.sha256()
    size = 0

    def fail_walk(error: OSError) -> None:
        raise error

    try:
        for root, dirs, files in os.walk(path, followlinks=False, onerror=fail_walk):
            dirs.sort()
            files.sort()
            for item in [Path(root), *(Path(root) / name for name in dirs + files)]:
                entry = item.lstat()
                if entry.st_dev != parent_device or stat.S_ISLNK(entry.st_mode):
                    return None
                if not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                    return None
                if stat.S_ISDIR(entry.st_mode) and item.is_mount():
                    return None
                if stat.S_ISREG(entry.st_mode):
                    size += entry.st_size
                digest.update(os.fsencode(str(item.relative_to(path))))
                digest.update(f"\0{entry.st_ino}\0{entry.st_size}\0{entry.st_mtime_ns}\0".encode())
    except OSError:
        return None
    return size, digest.hexdigest()


def _candidates(models_dir: Path) -> list[dict[str, Any]]:
    root = models_dir.parent / "decision-originals"
    if root.is_symlink():
        raise ValueError("Original storage path is invalid")
    if not root.exists():
        return []
    if not root.is_dir() or root.is_mount():
        raise ValueError("Original storage path is invalid")
    references = _referenced_originals(models_dir)
    root_device = root.stat().st_dev
    candidates = []
    for child in sorted(root.iterdir()):
        if not _ORIGINAL_NAME.fullmatch(child.name) or child.is_symlink() or not child.is_dir():
            continue
        resolved = child.resolve()
        if any(
            resolved == ref or resolved in ref.parents or ref in resolved.parents
            for ref in references
        ):
            continue
        snapshot = _tree_snapshot(child, root_device)
        if snapshot is not None:
            size, fingerprint = snapshot
            candidates.append({"name": child.name, "size_bytes": size, "fingerprint": fingerprint})
    return candidates


def list_unreferenced_originals(models_dir: Path) -> list[dict[str, Any]]:
    with _cleanup_locks(models_dir):
        return _candidates(models_dir)


def cleanup_unreferenced_originals(models_dir: Path, expected: list[dict[str, Any]]) -> int:
    """Recheck the preview under the install lock before moving and deleting anything."""
    if not expected or len(expected) > 100:
        raise ValueError("Select between 1 and 100 original directories")
    with _cleanup_locks(models_dir):
        candidates = {entry["name"]: entry for entry in _candidates(models_dir)}
        names = [entry.get("name") for entry in expected]
        if (
            any(not isinstance(name, str) or name not in candidates for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Cleanup targets changed; review them again")
        if any(entry != candidates[entry["name"]] for entry in expected):
            raise ValueError("Cleanup targets changed; review them again")

        root = models_dir.parent / "decision-originals"
        staging = Path(tempfile.mkdtemp(prefix=".cleanup-", dir=root))
        moved: list[str] = []
        try:
            for name in names:
                os.replace(root / name, staging / name)
                moved.append(name)
            # Verify the moved directories as well, so changes between the
            # preview scan and the rename do not lead to deleting new files.
            root_device = root.stat().st_dev
            for entry in expected:
                snapshot = _tree_snapshot(staging / entry["name"], root_device)
                if snapshot != (entry["size_bytes"], entry["fingerprint"]):
                    raise ValueError("Cleanup targets changed; review them again")
        except (OSError, ValueError) as exc:
            rollback_failed = False
            for name in reversed(moved):
                try:
                    destination = root / name
                    if destination.exists() or destination.is_symlink():
                        raise OSError("Original path was recreated")
                    os.replace(staging / name, destination)
                except OSError:
                    rollback_failed = True
            if not rollback_failed:
                staging.rmdir()
            else:
                raise OSError(f"Cleanup was incomplete; inspect {staging}") from exc
            raise
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise OSError(f"Cleanup was incomplete; inspect {staging}") from exc
        return len(names)
