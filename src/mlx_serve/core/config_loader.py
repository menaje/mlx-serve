"""YAML configuration file loader for mlx-serve."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path.home() / ".mlx-serve" / "config.yaml"


def build_default_config_dict(settings_obj: Any | None = None) -> dict[str, Any]:
    """Build the default config structure using the current effective settings."""
    if settings_obj is None:
        from mlx_serve.config import settings as settings_obj

    return {
        "server": {
            "host": settings_obj.host,
            "port": settings_obj.port,
        },
        "models": {
            "directory": str(settings_obj.models_dir),
            "preload": list(settings_obj.preload_models),
            "auto_download": settings_obj.auto_download,
            "auto_download_timeout": settings_obj.auto_download_timeout,
        },
        "cache": {
            "max_embedding_models": settings_obj.cache_max_embedding_models,
            "max_reranker_models": settings_obj.cache_max_reranker_models,
            "max_llm_models": settings_obj.cache_max_llm_models,
            "max_vlm_models": settings_obj.cache_max_vlm_models,
            "max_tts_models": settings_obj.cache_max_tts_models,
            "max_stt_models": settings_obj.cache_max_stt_models,
            "max_image_gen_models": settings_obj.cache_max_image_gen_models,
            "ttl_seconds": settings_obj.cache_ttl_seconds,
        },
        "batch": {
            "max_batch_size": settings_obj.batch_max_size,
            "max_wait_ms": settings_obj.batch_max_wait_ms,
            "max_concurrency_per_model": settings_obj.inference_max_concurrency_per_model,
            "max_queue_per_model": settings_obj.inference_max_queue_per_model,
            "queue_timeout_seconds": settings_obj.inference_queue_timeout_seconds,
        },
        "metrics": {
            "enabled": settings_obj.metrics_enabled,
            "port": settings_obj.metrics_port,
        },
        "service": {
            "auto_start": settings_obj.service_auto_start,
            "keep_alive": settings_obj.service_keep_alive,
        },
        "memory": {
            "guard_enabled": settings_obj.memory_guard_enabled,
            "poll_interval_seconds": settings_obj.memory_poll_interval_seconds,
            "process_limit_fraction": settings_obj.memory_process_limit_fraction,
            "min_available_fraction": settings_obj.memory_min_available_fraction,
        },
        "retrieval": {
            "worker_isolation_enabled": settings_obj.retrieval_worker_isolation_enabled,
            "worker_host": settings_obj.retrieval_worker_host,
            "worker_ready_timeout_seconds": settings_obj.retrieval_worker_ready_timeout_seconds,
            "worker_shutdown_timeout_seconds": settings_obj.retrieval_worker_shutdown_timeout_seconds,
            "clear_mlx_cache_after_request": (
                settings_obj.retrieval_clear_mlx_cache_after_request
            ),
        },
        "logging": {
            "level": settings_obj.log_level,
            "format": settings_obj.log_format,
            "debug_chat_request_bodies": settings_obj.debug_log_chat_request_bodies,
        },
    }


def render_default_config(settings_obj: Any | None = None) -> str:
    """Render a default config file using the current effective settings."""
    config = build_default_config_dict(settings_obj)
    body = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=False,
    )
    return (
        "# ~/.mlx-serve/config.yaml\n"
        "# MLX-Serve Configuration File\n\n"
        f"{body}"
    )


def ensure_default_config(config_path: Path | None = None) -> tuple[Path, bool]:
    """Create a default config file if it does not already exist."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return config_path, False

    config_path.write_text(render_default_config(), encoding="utf-8")
    logger.info(f"Created default config at {config_path}")
    return config_path, True


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary with configuration values, or empty dict if file not found.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.debug(f"Config file not found at {config_path}")
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return {}


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested config dict to match pydantic-settings field names.

    Example:
        {"server": {"host": "0.0.0.0"}} -> {"host": "0.0.0.0"}
        {"cache": {"max_embedding_models": 3}} -> {"cache_max_embedding_models": 3}
        {"models": {"preload": ["model1"]}} -> {"preload_models": ["model1"]}
    """
    result = {}

    # Mapping from YAML structure to Settings field names
    field_mappings = {
        # server section
        ("server", "host"): "host",
        ("server", "port"): "port",
        # models section
        ("models", "directory"): "models_dir",
        ("models", "preload"): "preload_models",
        ("models", "auto_download"): "auto_download",
        ("models", "auto_download_timeout"): "auto_download_timeout",
        # cache section
        ("cache", "max_embedding_models"): "cache_max_embedding_models",
        ("cache", "max_reranker_models"): "cache_max_reranker_models",
        ("cache", "ttl_seconds"): "cache_ttl_seconds",
        # batch section
        ("batch", "max_batch_size"): "batch_max_size",
        ("batch", "max_wait_ms"): "batch_max_wait_ms",
        ("batch", "max_concurrency_per_model"): "inference_max_concurrency_per_model",
        ("batch", "max_queue_per_model"): "inference_max_queue_per_model",
        ("batch", "queue_timeout_seconds"): "inference_queue_timeout_seconds",
        # memory section
        ("memory", "guard_enabled"): "memory_guard_enabled",
        ("memory", "poll_interval_seconds"): "memory_poll_interval_seconds",
        ("memory", "process_limit_fraction"): "memory_process_limit_fraction",
        ("memory", "min_available_fraction"): "memory_min_available_fraction",
        # retrieval section
        ("retrieval", "worker_isolation_enabled"): "retrieval_worker_isolation_enabled",
        ("retrieval", "worker_host"): "retrieval_worker_host",
        ("retrieval", "worker_ready_timeout_seconds"): "retrieval_worker_ready_timeout_seconds",
        (
            "retrieval",
            "worker_shutdown_timeout_seconds",
        ): "retrieval_worker_shutdown_timeout_seconds",
        (
            "retrieval",
            "clear_mlx_cache_after_request",
        ): "retrieval_clear_mlx_cache_after_request",
        # metrics section
        ("metrics", "enabled"): "metrics_enabled",
        ("metrics", "port"): "metrics_port",
        # service section
        ("service", "auto_start"): "service_auto_start",
        ("service", "keep_alive"): "service_keep_alive",
        # logging section
        ("logging", "level"): "log_level",
        ("logging", "format"): "log_format",
        ("logging", "debug_chat_request_bodies"): "debug_log_chat_request_bodies",
    }

    for section_name, section_value in config.items():
        if isinstance(section_value, dict):
            for key, value in section_value.items():
                mapping_key = (section_name, key)
                if mapping_key in field_mappings:
                    result[field_mappings[mapping_key]] = value
                else:
                    # Use underscore-joined name as fallback
                    result[f"{section_name}_{key}"] = value
        else:
            # Top-level value
            result[section_name] = section_value

    return result


def get_config_values(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load and flatten config from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Flattened dictionary with configuration values.
    """
    raw_config = load_yaml_config(config_path)
    return flatten_config(raw_config)


def get_example_config() -> str:
    """Return example configuration file content."""
    return """\
# ~/.mlx-serve/config.yaml
# MLX-Serve Configuration File

server:
  host: "0.0.0.0"
  port: 8000

models:
  directory: ~/.mlx-serve/models
  preload:
    - Qwen3-Embedding-0.6B
  auto_download: true

cache:
  max_embedding_models: 3
  max_reranker_models: 2
  ttl_seconds: 1800

batch:
  max_batch_size: 32
  max_wait_ms: 50
  max_concurrency_per_model: 1
  max_queue_per_model: 8
  queue_timeout_seconds: 30

metrics:
  enabled: true
  port: 9090

service:
  auto_start: true
  keep_alive: true

memory:
  guard_enabled: true
  poll_interval_seconds: 2.0
  process_limit_fraction: 0.75
  min_available_fraction: 0.10

retrieval:
  worker_isolation_enabled: true
  worker_host: 127.0.0.1
  worker_ready_timeout_seconds: 30.0
  worker_shutdown_timeout_seconds: 5.0
  clear_mlx_cache_after_request: true

logging:
  level: INFO
  format: json  # text | json
  debug_chat_request_bodies: false
"""
