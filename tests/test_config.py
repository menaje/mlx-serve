"""Tests for configuration module."""

import tempfile
from pathlib import Path

import yaml


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_load_yaml_config_missing_file(self):
        """Test loading config when file doesn't exist."""
        from mlx_serve.core.config_loader import load_yaml_config

        config = load_yaml_config(Path("/nonexistent/path/config.yaml"))
        assert config == {}

    def test_load_yaml_config_valid(self):
        """Test loading valid YAML config."""
        from mlx_serve.core.config_loader import load_yaml_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"server": {"host": "127.0.0.1", "port": 9000}}, f)
            f.flush()

            config = load_yaml_config(Path(f.name))
            assert config["server"]["host"] == "127.0.0.1"
            assert config["server"]["port"] == 9000

    def test_flatten_config(self):
        """Test flattening nested config."""
        from mlx_serve.core.config_loader import flatten_config

        nested = {
            "server": {"host": "0.0.0.0", "port": 8000},
            "cache": {"max_embedding_models": 5},
            "models": {"preload": ["model1", "model2"]},
            "batch": {"max_queue_per_model": 4, "queue_timeout_seconds": 12},
            "memory": {"guard_enabled": True, "min_available_fraction": 0.05},
            "logging": {"debug_chat_request_bodies": True},
        }

        flat = flatten_config(nested)
        assert flat["host"] == "0.0.0.0"
        assert flat["port"] == 8000
        assert flat["cache_max_embedding_models"] == 5
        assert flat["preload_models"] == ["model1", "model2"]
        assert flat["inference_max_queue_per_model"] == 4
        assert flat["inference_queue_timeout_seconds"] == 12
        assert flat["memory_guard_enabled"] is True
        assert flat["memory_min_available_fraction"] == 0.05
        assert flat["debug_log_chat_request_bodies"] is True

    def test_get_example_config(self):
        """Test example config generation."""
        from mlx_serve.core.config_loader import get_example_config

        example = get_example_config()
        assert "server:" in example
        assert "models:" in example
        assert "cache:" in example
        assert "metrics:" in example


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        from mlx_serve.config import Settings

        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "INFO"
        assert settings.cache_max_embedding_models == 3
        assert settings.inference_max_concurrency_per_model == 1
        assert settings.inference_max_queue_per_model == 8
        assert settings.inference_queue_timeout_seconds == 30.0
        assert settings.memory_guard_enabled is True
        assert settings.debug_log_chat_request_bodies is False

    def test_preload_models_parsing(self):
        """Test parsing preload_models from string."""
        from mlx_serve.config import Settings

        settings = Settings(preload_models="model1, model2, model3")
        assert settings.preload_models == ["model1", "model2", "model3"]

    def test_models_dir_expansion(self):
        """Test ~ expansion in models_dir."""
        from mlx_serve.config import Settings

        settings = Settings(models_dir="~/test/models")
        assert "~" not in str(settings.models_dir)
        assert settings.models_dir.is_absolute()
