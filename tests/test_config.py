"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
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
        }

        flat = flatten_config(nested)
        assert flat["host"] == "0.0.0.0"
        assert flat["port"] == 8000
        assert flat["cache_max_embedding_models"] == 5
        assert flat["preload_models"] == ["model1", "model2"]

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
