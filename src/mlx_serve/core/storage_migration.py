"""Copy model storage without removing or overwriting the original assets."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from mlx_serve.core.managed_config import save_settings
from mlx_serve.core.storage_access import model_storage_available, model_volume_mounted


@dataclass(frozen=True)
class _Entry:
    source: Path
    relative: Path
    identity: tuple[int, int, int, int, int, int]


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _migration_locks(models: Path) -> Iterator[None]:
    """Use the same lock order as preparation and default-model updates."""
    if not model_storage_available(models) or models.is_symlink():
        raise ValueError("Model storage is unavailable or is a symbolic link")
    lock_dir = models / ".decision-locks"
    if lock_dir.is_symlink():
        raise ValueError("Decision preparation lock directory is invalid")
    lock_dir.mkdir(exist_ok=True)
    paths = (
        lock_dir / "d1-preparation.lock",
        models.parent / ".decision-metadata.lock",
        models.parent / ".default-models.lock",
    )
    with ExitStack() as stack:
        for path in paths:
            if path.is_symlink():
                raise ValueError("Model storage lock is invalid")
            lock = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Model storage is being modified by another process") from exc
            stack.callback(fcntl.flock, lock.fileno(), fcntl.LOCK_UN)
        if not model_storage_available(models):
            raise ValueError("Model storage is unavailable")
        yield


def _paths(source: Path, requested: Path) -> tuple[Path, Path]:
    requested = requested.expanduser()
    if not source.is_absolute() or not requested.is_absolute():
        raise ValueError("Model directory must be an absolute path")
    if source.is_symlink():
        raise ValueError("Model storage is unavailable or is a symbolic link")
    source = source.resolve()
    destination = requested.resolve()
    if not model_volume_mounted(source) or not model_volume_mounted(destination):
        raise ValueError("Model storage volume is unavailable; reconnect it first")
    if destination.name in {"decision-originals", "metadata.json", "default-models.json"}:
        raise ValueError("Choose a separate model directory")
    new_parent = destination.parent
    if new_parent.exists() or new_parent.is_symlink():
        raise ValueError("The destination parent directory already exists")
    if not new_parent.parent.is_dir() or not os.access(new_parent.parent, os.W_OK | os.X_OK):
        raise ValueError("The destination parent must be a new folder inside a writable folder")
    if new_parent.is_relative_to(source.parent) or source.parent.is_relative_to(new_parent):
        raise ValueError("Choose a destination outside the existing model storage")
    return source, destination


def _snapshot(source: Path, destination: Path) -> list[_Entry]:
    if not model_storage_available(source):
        raise ValueError("Model storage is unavailable")
    device = source.stat().st_dev
    roots = (
        (source, Path(destination.name)),
        (source.parent / "decision-originals", Path("decision-originals")),
        (source.parent / "metadata.json", Path("metadata.json")),
        (source.parent / "default-models.json", Path("default-models.json")),
    )
    entries: list[_Entry] = []

    def visit(path: Path, relative: Path) -> None:
        info = path.lstat()
        if info.st_dev != device or stat.S_ISLNK(info.st_mode):
            raise ValueError("Model storage contains a link or another mounted volume")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError("Model storage contains an unsupported file")
        if stat.S_ISDIR(info.st_mode) and path.is_mount():
            raise ValueError("Model storage contains a link or another mounted volume")
        entries.append(_Entry(path, relative, _identity(info)))
        if stat.S_ISDIR(info.st_mode):
            for item in sorted(path.iterdir(), key=lambda value: value.name):
                if path == source and item.name == ".decision-locks":
                    continue
                if path == source and item.name.startswith(".decision-prepare-"):
                    raise ValueError("An unfinished preparation directory needs review")
                visit(item, relative / item.name)

    for path, relative in roots:
        if path.exists() or path.is_symlink():
            visit(path, relative)
    return entries


def _validated_registry(source: Path) -> dict:
    path = source.parent / "metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(item, dict) for item in data.values()):
            raise ValueError
        for model in data.values():
            original = model.get("original_path")
            if original is not None and (
                not isinstance(original, str) or not Path(original).is_absolute()
            ):
                raise ValueError
        return data
    except (OSError, ValueError) as exc:
        raise ValueError("Model registry could not be checked") from exc


def _rewrite_registry(registry: dict, source: Path, destination: Path) -> dict:
    for model in registry.values():
        original = model.get("original_path")
        if not original:
            continue
        path = Path(original).resolve()
        for old, new in (
            (source, destination),
            (source.parent / "decision-originals", destination.parent / "decision-originals"),
        ):
            if path.is_relative_to(old):
                model["original_path"] = str(new / path.relative_to(old))
                break
    return registry


def _copy_file(
    entry: _Entry, target: Path, cancelled: threading.Event,
    progress: Callable[[int], None],
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(entry.source, flags)
    try:
        before = os.fstat(source_fd)
        if _identity(before) != entry.identity:
            raise ValueError("Model storage changed during migration")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        with (
            os.fdopen(target_fd, "wb") as output,
            os.fdopen(source_fd, "rb", closefd=False) as input_file,
        ):
            while chunk := input_file.read(4 * 1024 * 1024):
                if cancelled.is_set():
                    raise InterruptedError("Model storage copy was interrupted")
                output.write(chunk)
                digest.update(chunk)
                progress(len(chunk))
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), stat.S_IMODE(before.st_mode))
        if _identity(os.fstat(source_fd)) != entry.identity:
            raise ValueError("Model storage changed during migration")
    finally:
        os.close(source_fd)
    copied = hashlib.sha256()
    with target.open("rb") as file:
        while chunk := file.read(4 * 1024 * 1024):
            if cancelled.is_set():
                raise InterruptedError("Model storage copy was interrupted")
            copied.update(chunk)
    if copied.digest() != digest.digest():
        raise ValueError("Copied model files failed verification")


def _copy_size(source: Path, destination: Path) -> int:
    entries = _snapshot(source, destination)
    _validated_registry(source)
    return sum(item.identity[3] for item in entries if stat.S_ISREG(item.identity[2]))


def preview_storage_migration(source: Path, requested: Path) -> dict[str, object]:
    """Describe the source, destination and available space before copying."""
    source, destination = _paths(source, requested)
    with _migration_locks(source):
        size = _copy_size(source, destination)
        free = shutil.disk_usage(destination.parent.parent).free
        return {
            "source": str(source), "destination": str(destination),
            "size_bytes": size, "free_bytes": free,
        }


def _write_registry(path: Path, registry: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
        file.flush()
        os.fsync(file.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def migrate_storage(
    source: Path, requested: Path, *, cancelled: threading.Event,
    update: Callable[[str, int, int], None],
    save: Callable[..., None] = save_settings,
) -> dict[str, object]:
    """Copy and verify first; switch configuration only after all checks succeed.

    Neither existing source files nor an incomplete destination are deleted.
    """
    source, destination = _paths(source, requested)
    with _migration_locks(source):
        before = _snapshot(source, destination)
        registry = _validated_registry(source)
        defaults = source.parent / "default-models.json"
        if defaults.exists():
            try:
                if not isinstance(json.loads(defaults.read_text(encoding="utf-8")), dict):
                    raise ValueError
            except (OSError, ValueError) as exc:
                raise ValueError("Default model settings could not be read") from exc
        total = sum(item.identity[3] for item in before if stat.S_ISREG(item.identity[2]))
        required = total + max(16 * 1024 * 1024, total // 100)
        if shutil.disk_usage(destination.parent.parent).free < required:
            raise ValueError("Not enough free disk space for the model copy")
        if cancelled.is_set():
            raise InterruptedError("Model storage copy was interrupted")
        destination.parent.mkdir(mode=0o700)
        copied = 0
        copied_files: dict[Path, tuple[int, int, int, int, int, int]] = {}

        def report(size: int) -> None:
            nonlocal copied
            copied += size
            update("copying", copied, total)

        update("copying", copied, total)
        for entry in before:
            if cancelled.is_set():
                raise InterruptedError("Model storage copy was interrupted")
            if not model_volume_mounted(source) or not model_volume_mounted(destination):
                raise ValueError("Model storage volume is unavailable; reconnect it first")
            target = destination.parent / entry.relative
            if stat.S_ISDIR(entry.identity[2]):
                # The source may be read-only. Keep the private copy writable
                # until its contents and integrity checks have completed.
                target.mkdir(mode=0o700)
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
                fd = os.open(target, flags)
                try:
                    os.fchmod(fd, 0o700)
                finally:
                    os.close(fd)
            elif entry.source == source.parent / "metadata.json":
                _write_registry(target, _rewrite_registry(registry, source, destination))
                report(entry.identity[3])
            else:
                _copy_file(entry, target, cancelled, report)
            if stat.S_ISREG(entry.identity[2]):
                copied_files[entry.relative] = _identity(target.lstat())
        update("checking", copied, total)
        if _snapshot(source, destination) != before:
            raise ValueError("Model storage changed during migration")
        target_lock = destination / ".decision-locks"
        if target_lock.exists() or target_lock.is_symlink():
            raise ValueError("Copied model files changed during migration")
        copied_entries = _snapshot(destination, destination)
        if {entry.relative for entry in copied_entries} != {entry.relative for entry in before}:
            raise ValueError("Copied model files changed during migration")
        if any(
            entry.identity != copied_files.get(entry.relative)
            for entry in copied_entries if stat.S_ISREG(entry.identity[2])
        ):
            raise ValueError("Copied model files changed during migration")
        if cancelled.is_set():
            raise InterruptedError("Model storage copy was interrupted")
        if not model_volume_mounted(source) or not model_volume_mounted(destination):
            raise ValueError("Model storage volume is unavailable; reconnect it first")
        for entry in reversed(before):
            if stat.S_ISDIR(entry.identity[2]):
                _sync_directory(destination.parent / entry.relative)
        _sync_directory(destination.parent)
        _sync_directory(destination.parent.parent)
        if cancelled.is_set():
            raise InterruptedError("Model storage copy was interrupted")
        if _snapshot(destination, destination) != copied_entries:
            raise ValueError("Copied model files changed during migration")
        if target_lock.exists() or target_lock.is_symlink():
            raise ValueError("Copied model files changed during migration")
        if not model_volume_mounted(source) or not model_volume_mounted(destination):
            raise ValueError("Model storage volume is unavailable; reconnect it first")
        copied_directories = {
            entry.relative: entry.identity
            for entry in copied_entries if stat.S_ISDIR(entry.identity[2])
        }
        for entry in reversed(before):
            if not stat.S_ISDIR(entry.identity[2]):
                continue
            target = destination.parent / entry.relative
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
            fd = os.open(target, flags)
            try:
                if _identity(os.fstat(fd)) != copied_directories.get(entry.relative):
                    raise ValueError("Copied model files changed during migration")
                os.fchmod(fd, stat.S_IMODE(entry.identity[2]))
                os.fsync(fd)
            finally:
                os.close(fd)
        update("applying", copied, total)
        save({"models_dir": str(destination)}, expected_models_dir=source)
        return {"source": str(source), "destination": str(destination), "size_bytes": total}
