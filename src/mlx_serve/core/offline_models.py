"""Inspect and remove registered model copies while the app's server is stopped."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mlx_serve.core.managed_originals import _cleanup_locks, _tree_snapshot
from mlx_serve.core.storage_access import model_volume_mounted

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TYPES = {"decision", "embedding", "reranker", "llm", "vlm", "tts", "stt", "image_gen"}


def _registry(models_dir: Path) -> dict[str, Any]:
    path = models_dir.parent / "metadata.json"
    if path.is_symlink():
        raise ValueError("Model registry could not be checked")
    if not path.exists():
        if any(p.name != ".decision-locks" for p in models_dir.iterdir()):
            raise ValueError("Model registry could not be checked")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(not isinstance(v, dict) for v in value.values()):
            raise ValueError
        return value
    except (OSError, ValueError) as exc:
        raise ValueError("Model registry could not be checked") from exc


def _record(models_dir: Path, registry: dict[str, Any], name: str, kind: str) -> Path:
    if not _NAME.fullmatch(name) or kind not in _TYPES:
        raise ValueError("Model name or type is invalid")
    model = registry.get(name)
    if not isinstance(model, dict) or model.get("type") != kind:
        raise ValueError("The selected model is not installed")
    target = models_dir / name
    if target.is_symlink() or not target.is_dir() or target.is_mount():
        raise ValueError("Model files could not be safely checked")
    return target


def _paths(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _paths(item)
    elif isinstance(value, str) and value.startswith("/"):
        yield Path(value).resolve()


def _preview(models_dir: Path, registry: dict[str, Any], name: str, kind: str) -> dict[str, Any]:
    target = _record(models_dir, registry, name, kind)
    resolved = target.resolve()
    for other, record in registry.items():
        if other != name and any(
            path == resolved or path in resolved.parents or resolved in path.parents
            for path in _paths(record)
        ):
            raise ValueError("Another model references these files")
    snapshot = _tree_snapshot(target, models_dir.stat().st_dev)
    if snapshot is None:
        raise ValueError("Model files could not be safely checked")
    size, tree_fingerprint = snapshot
    fingerprint = _removal_fingerprint(tree_fingerprint, registry[name])
    return {"name": name, "type": kind, "size_bytes": size, "fingerprint": fingerprint}


def _removal_fingerprint(tree_fingerprint: str, record: dict[str, Any]) -> str:
    value = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((tree_fingerprint + value).encode("utf-8")).hexdigest()


def offline_model_list(models_dir: Path) -> dict[str, Any]:
    if not model_volume_mounted(models_dir):
        raise ValueError("Model storage is unavailable")
    if not models_dir.exists():
        return {"models": [], "storage": {"state": "available"}}
    with _cleanup_locks(models_dir):
        registry = _registry(models_dir)
        models = []
        for name, record in sorted(registry.items()):
            kind = record.get("type")
            if not isinstance(name, str) or not _NAME.fullmatch(name) or kind not in _TYPES:
                continue
            path = models_dir / name
            if path.is_symlink() or not path.is_dir():
                continue
            size = record.get("size")
            models.append({
                "name": name, "type": kind, "source": record.get("hf_repo", ""),
                "metadata": record, "size_bytes": size if isinstance(size, int) else 0,
                "file_state": "installed", "load_state": "unknown", "request_state": "unknown",
            })
        return {"models": models, "storage": {"state": "available"}}


def preview_model_removal(models_dir: Path, name: str, kind: str) -> dict[str, Any]:
    with _cleanup_locks(models_dir):
        return _preview(models_dir, _registry(models_dir), name, kind)


def remove_offline_model(models_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Remove only a reviewed unchanged registered copy; keep other source caches."""
    with _cleanup_locks(models_dir):
        registry = _registry(models_dir)
        current = _preview(models_dir, registry, expected["name"], expected["type"])
        if current != expected:
            raise ValueError("Removal target changed; review it again")
        target = models_dir / expected["name"]
        staging = Path(tempfile.mkdtemp(prefix=".model-removal-", dir=models_dir))
        moved = staging / "model"
        try:
            os.replace(target, moved)
            moved_snapshot = _tree_snapshot(moved, models_dir.stat().st_dev)
            if (moved_snapshot is None or moved_snapshot[0] != expected["size_bytes"]
                    or _removal_fingerprint(moved_snapshot[1], registry[expected["name"]])
                    != expected["fingerprint"]):
                raise ValueError("Removal target changed; review it again")
            del registry[expected["name"]]
            temp = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=models_dir.parent, delete=False,
                ) as file:
                    temp = Path(file.name)
                    os.fchmod(file.fileno(), 0o600)
                    json.dump(registry, file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temp, models_dir.parent / "metadata.json")
                temp = None
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        except Exception as exc:
            try:
                if moved.exists():
                    if target.exists() or target.is_symlink():
                        raise OSError("Model path was recreated")
                    os.replace(moved, target)
                staging.rmdir()
            except OSError:
                raise OSError(f"Removal was incomplete; inspect {staging}") from exc
            raise
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise OSError(f"Removal was incomplete; inspect {staging}") from exc
        if expected["type"] in ("decision", "embedding", "reranker"):
            from mlx_serve.core.default_models import _update_selection

            with contextlib.suppress(OSError, ValueError):
                _update_selection(models_dir, expected["type"], None,
                                  expected_name=expected["name"], ignore_changed=True)
        return {"removed": expected["name"], "size_bytes": expected["size_bytes"]}
