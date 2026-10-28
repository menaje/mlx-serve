"""Validated, atomic configuration editing for local management clients."""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import DotEnvSettingsSource, EnvSettingsSource

from mlx_serve.config import Settings
from mlx_serve.core.config_loader import DEFAULT_CONFIG_PATH, flatten_config
from mlx_serve.core.storage_access import model_volume_mounted

# Only options that are wired into the existing engine are exposed for editing.
EDITABLE_FIELDS: dict[str, tuple[str, str]] = {
    "host": ("server", "host"),
    "port": ("server", "port"),
    "models_dir": ("models", "directory"),
    "preload_models": ("models", "preload"),
    "cache_ttl_seconds": ("cache", "ttl_seconds"),
    "cache_max_embedding_models": ("cache", "max_embedding_models"),
    "cache_max_reranker_models": ("cache", "max_reranker_models"),
    "cache_max_decision_models": ("cache", "max_decision_models"),
    "inference_max_concurrency_per_model": ("batch", "max_concurrency_per_model"),
    "inference_max_queue_per_model": ("batch", "max_queue_per_model"),
    "inference_queue_timeout_seconds": ("batch", "queue_timeout_seconds"),
    "embedding_batch_max_texts": ("batch", "max_embedding_texts"),
    "rerank_batch_max_documents": ("batch", "rerank_batch_max_documents"),
    "rerank_batch_max_tokens": ("batch", "rerank_batch_max_tokens"),
    "memory_guard_enabled": ("memory", "guard_enabled"),
    "memory_process_limit_fraction": ("memory", "process_limit_fraction"),
    "memory_min_available_fraction": ("memory", "min_available_fraction"),
    "memory_load_headroom_fraction": ("memory", "load_headroom_fraction"),
    "decision_prepare_timeout_seconds": ("decision", "prepare_timeout_seconds"),
    "decision_prepare_max_memory_bytes": ("decision", "prepare_max_memory_bytes"),
    "decision_prepare_max_disk_bytes": ("decision", "prepare_max_disk_bytes"),
    "log_level": ("logging", "level"),
    "debug_log_chat_request_bodies": ("logging", "debug_chat_request_bodies"),
}


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        # YAML parser exceptions can contain the user's configuration contents.
        raise ValueError("Configuration is not valid YAML") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return data


def _defaults() -> dict[str, Any]:
    # model_construct intentionally avoids all environment/YAML sources.
    return Settings.model_construct().model_dump()


def _settings(values: dict[str, Any]) -> Settings:
    # Supplying every field as an initial value prevents unrelated process
    # environment or the global YAML path from changing this validation.
    return Settings(**{**_defaults(), **values})


def _source_values() -> tuple[dict[str, Any], dict[str, Any]]:
    environment = EnvSettingsSource(Settings)()
    dotenv = DotEnvSettingsSource(Settings, env_file=Settings.model_config.get("env_file"))()
    return environment, dotenv


def _validate_directory(value: Path) -> None:
    if not value.is_absolute():
        raise ValueError("Model directory must be an absolute path")
    if not model_volume_mounted(value):
        raise ValueError("Model storage volume is unavailable; reconnect it first")
    candidate = value
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or not os.access(candidate, os.W_OK | os.X_OK):
        raise ValueError("Model directory or its existing parent is not writable")
    if value.exists() and not value.is_dir():
        raise ValueError("Model directory is not a directory")


def settings_snapshot(
    path: Path = DEFAULT_CONFIG_PATH, *, applied: Settings | None = None
) -> dict[str, Any]:
    """Return saved, prospective and actual values without conflating them."""
    raw = _read_config(path)
    yaml_values = flatten_config(raw)
    saved = _settings(yaml_values).model_dump(mode="json")
    environment, dotenv = _source_values()
    prospective = _settings({**yaml_values, **dotenv, **environment}).model_dump(mode="json")
    actual = applied.model_dump(mode="json") if applied is not None else None
    schema = Settings.model_json_schema().get("properties", {})
    fields = {}
    for name in EDITABLE_FIELDS:
        if name in environment:
            source = "environment"
        elif name in dotenv:
            source = "dotenv"
        elif name in yaml_values:
            source = "yaml"
        else:
            source = "default"
        fields[name] = {
            "saved": saved[name],
            "effective_on_restart": prospective[name],
            "applied": actual[name] if actual is not None else None,
            "source": source,
            "restart_required": actual is not None and prospective[name] != actual[name],
            "schema": schema.get(name, {}),
        }
    return {"path": str(path), "fields": fields}


def _atomic_write(path: Path, raw: dict[str, Any]) -> None:
    text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    try:
        # Configuration may contain user-supplied values, so use private perms.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_settings(changes: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Validate the complete candidate before replacing the original file."""
    unknown = changes.keys() - EDITABLE_FIELDS.keys()
    if unknown:
        raise ValueError(f"Unsupported settings: {', '.join(sorted(unknown))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".config.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            raw = _read_config(path)
            for name, value in changes.items():
                section, key = EDITABLE_FIELDS[name]
                existing = raw.setdefault(section, {})
                if not isinstance(existing, dict):
                    raise ValueError(f"Configuration section '{section}' must be a mapping")
                existing[key] = value
            candidate = _settings(flatten_config(raw))
            if candidate.host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "::"):
                raise ValueError("Host must be a loopback or wildcard address")
            _validate_directory(candidate.models_dir)
            # Save the normalized types, but preserve unrelated CLI settings.
            normalized = candidate.model_dump(mode="json")
            for name in changes:
                section, key = EDITABLE_FIELDS[name]
                raw[section][key] = normalized[name]
            _atomic_write(path, raw)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def initialize_app_config(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Create a local-only config for a new app install; keep CLI config intact."""
    # A previous CLI install without a YAML file still has meaningful defaults.
    # Do not silently change its bind address merely because the app is new.
    existing_install = (
        (path.parent / "metadata.json").exists()
        or (path.parent / "models").exists()
        or any(path.parent.glob("mlx-serve-*.pid"))
        or Path("/opt/homebrew/bin/mlx-serve").exists()
        or Path("/usr/local/bin/mlx-serve").exists()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".config.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists() or existing_install:
                return False
            _atomic_write(path, {"server": {"host": "127.0.0.1", "port": 8000}})
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
