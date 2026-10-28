"""Per-type default models shared by the gateway and inference workers."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from mlx_serve.config import settings
from mlx_serve.core.storage_access import model_storage_available

DEFAULT_TYPES = frozenset({"decision", "embedding", "reranker"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IDENTITY_FIELDS = ("type", "hf_repo", "revision", "precision", "modified_at")


class DefaultSelectionChanged(ValueError):
    """A different model was selected before the requested clear."""


def _selections_path(models_dir: Path) -> Path:
    return models_dir.parent / "default-models.json"


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Default model settings could not be read") from exc
    if not isinstance(value, dict):
        raise ValueError("Default model settings are invalid")
    return value


def _registered_model(models_dir: Path, model_type: str, name: str) -> dict[str, Any] | None:
    if not _NAME.fullmatch(name) or not model_storage_available(models_dir):
        return None
    registry = _read_mapping(models_dir.parent / "metadata.json")
    model = registry.get(name)
    if not isinstance(model, dict) or model.get("type") != model_type:
        return None
    if not (models_dir / name).is_dir():
        return None
    return model


def _identity(model: dict[str, Any]) -> dict[str, Any]:
    return {key: model.get(key) for key in _IDENTITY_FIELDS}


def default_model(model_type: str, models_dir: Path | None = None) -> str | None:
    """Return a still-registered selection, never a stale or inaccessible one."""
    directory = models_dir if models_dir is not None else settings.models_dir
    if model_type not in DEFAULT_TYPES or not model_storage_available(directory):
        return None
    try:
        selection = _read_mapping(_selections_path(directory)).get(model_type)
        if not isinstance(selection, dict):
            return None
        name = selection.get("name")
        if not isinstance(name, str):
            return None
        model = _registered_model(directory, model_type, name)
        if model is not None and selection.get("identity") == _identity(model):
            return name
    except ValueError:
        pass
    return None


@contextmanager
def _decision_registry_lock(directory: Path) -> Iterator[None]:
    with (directory.parent / ".decision-metadata.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _update_selection(
    directory: Path, model_type: str, name: str | None, *, expected_name: str | None = None,
    ignore_changed: bool = False,
) -> None:
    path = _selections_path(directory)
    with (path.parent / ".default-models.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        temporary: str | None = None
        try:
            selections = _read_mapping(path)
            if name is None:
                previous = selections.get(model_type)
                if expected_name is not None and (
                    not isinstance(previous, dict) or previous.get("name") != expected_name
                ):
                    if ignore_changed:
                        return
                    raise DefaultSelectionChanged("This model is not the current default")
                selections.pop(model_type, None)
            else:
                model = _registered_model(directory, model_type, name)
                if model is None:
                    raise ValueError("The selected model is not installed")
                selections[model_type] = {"name": name, "identity": _identity(model)}
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as file:
                temporary = file.name
                os.fchmod(file.fileno(), 0o600)
                json.dump(selections, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            fcntl.flock(lock, fcntl.LOCK_UN)


def set_default_model(
    model_type: str, name: str | None, models_dir: Path | None = None, *,
    expected_name: str | None = None,
) -> None:
    """Update one selection atomically, preserving selections for other types."""
    directory = models_dir if models_dir is not None else settings.models_dir
    if model_type not in DEFAULT_TYPES:
        raise ValueError("Default selection is not supported for this model type")
    if not model_storage_available(directory):
        raise ValueError("Model storage is unavailable")
    publication = _decision_registry_lock(directory) if model_type == "decision" else nullcontext()
    with publication:
        _update_selection(directory, model_type, name, expected_name=expected_name)


def clear_deleted_decision_default(name: str, models_dir: Path) -> None:
    """Clear a deleted decision default while holding the decision registry lock."""
    _update_selection(models_dir, "decision", None, expected_name=name, ignore_changed=True)


def request_model_default(data: Any, model_type: str) -> Any:
    """Supply the default only when the caller did not provide a model field."""
    if isinstance(data, dict) and "model" not in data:
        name = default_model(model_type)
        if name is not None:
            return {**data, "model": name}
    return data
