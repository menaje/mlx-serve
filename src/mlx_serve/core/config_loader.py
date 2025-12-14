"""YAML configuration file loader for mlx-serve."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path.home() / ".mlx-serve" / "config.yaml"


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
        # metrics section
        ("metrics", "enabled"): "metrics_enabled",
        ("metrics", "port"): "metrics_port",
        # logging section
        ("logging", "level"): "log_level",
        ("logging", "format"): "log_format",
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

metrics:
  enabled: true
  port: 9090

logging:
  level: INFO
  format: json  # text | json
"""
