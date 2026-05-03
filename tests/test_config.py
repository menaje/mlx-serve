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
            "batch": {
                "max_queue_per_model": 4,
                "queue_timeout_seconds": 12,
                "max_embedding_texts": 16,
                "rerank_batch_max_documents": 3,
                "rerank_batch_max_tokens": 512,
            },
            "service": {"auto_start": False, "keep_alive": False},
            "memory": {
                "guard_enabled": True,
                "min_available_fraction": 0.05,
                "generation_guard_enabled": False,
                "generation_kv_bytes_per_token": 1024,
                "vlm_image_tokens_per_image": 256,
                "remote_model_estimates_bytes": {"remote-model": 2048},
            },
            "generation": {
                "worker_isolation_enabled": True,
                "worker_mode": "model",
                "worker_host": "127.0.0.2",
                "worker_ready_timeout_seconds": 11.0,
                "worker_shutdown_timeout_seconds": 3.0,
                "worker_idle_timeout_seconds": 7.0,
                "clear_mlx_cache_after_request": False,
                "kv_bits": 4,
                "kv_group_size": 32,
                "quantized_kv_start": 128,
                "prompt_cache_enabled": False,
                "prompt_cache_checkpoint_enabled": True,
                "prompt_cache_max_entries": 2,
                "prompt_cache_min_tokens": 16,
            },
            "retrieval": {"clear_mlx_cache_after_request": False},
            "logging": {"debug_chat_request_bodies": True},
        }

        flat = flatten_config(nested)
        assert flat["host"] == "0.0.0.0"
        assert flat["port"] == 8000
        assert flat["cache_max_embedding_models"] == 5
        assert flat["preload_models"] == ["model1", "model2"]
        assert flat["inference_max_queue_per_model"] == 4
        assert flat["inference_queue_timeout_seconds"] == 12
        assert flat["embedding_batch_max_texts"] == 16
        assert flat["rerank_batch_max_documents"] == 3
        assert flat["rerank_batch_max_tokens"] == 512
        assert flat["service_auto_start"] is False
        assert flat["service_keep_alive"] is False
        assert flat["memory_guard_enabled"] is True
        assert flat["memory_min_available_fraction"] == 0.05
        assert flat["memory_generation_guard_enabled"] is False
        assert flat["memory_generation_kv_bytes_per_token"] == 1024
        assert flat["memory_vlm_image_tokens_per_image"] == 256
        assert flat["memory_remote_model_estimates_bytes"] == {"remote-model": 2048}
        assert flat["generation_worker_isolation_enabled"] is True
        assert flat["generation_worker_mode"] == "model"
        assert flat["generation_worker_host"] == "127.0.0.2"
        assert flat["generation_worker_ready_timeout_seconds"] == 11.0
        assert flat["generation_worker_shutdown_timeout_seconds"] == 3.0
        assert flat["generation_worker_idle_timeout_seconds"] == 7.0
        assert flat["generation_clear_mlx_cache_after_request"] is False
        assert flat["generation_kv_bits"] == 4
        assert flat["generation_kv_group_size"] == 32
        assert flat["generation_quantized_kv_start"] == 128
        assert flat["generation_prompt_cache_enabled"] is False
        assert flat["generation_prompt_cache_checkpoint_enabled"] is True
        assert flat["generation_prompt_cache_max_entries"] == 2
        assert flat["generation_prompt_cache_min_tokens"] == 16
        assert flat["retrieval_clear_mlx_cache_after_request"] is False
        assert flat["debug_log_chat_request_bodies"] is True

    def test_get_example_config(self):
        """Test example config generation."""
        from mlx_serve.core.config_loader import get_example_config

        example = get_example_config()
        assert "server:" in example
        assert "models:" in example
        assert "cache:" in example
        assert "metrics:" in example
        assert "service:" in example

    def test_render_default_config(self):
        """Test rendering the default config from effective settings."""
        from mlx_serve.config import Settings
        from mlx_serve.core.config_loader import render_default_config

        config_text = render_default_config(Settings())
        assert "server:" in config_text
        assert "port: 8000" in config_text
        assert "service:" in config_text
        assert "auto_start: true" in config_text
        assert "keep_alive: true" in config_text

    def test_ensure_default_config_creates_file(self, tmp_path):
        """Test creating a default config file when missing."""
        from mlx_serve.core.config_loader import ensure_default_config

        config_path = tmp_path / "config.yaml"
        created_path, created = ensure_default_config(config_path)

        assert created is True
        assert created_path == config_path
        assert config_path.exists()
        content = config_path.read_text()
        assert "server:" in content
        assert "service:" in content

    def test_ensure_default_config_does_not_overwrite(self, tmp_path):
        """Test preserving an existing config file."""
        from mlx_serve.core.config_loader import ensure_default_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text("server:\n  port: 9000\n", encoding="utf-8")

        created_path, created = ensure_default_config(config_path)

        assert created is False
        assert created_path == config_path
        assert config_path.read_text(encoding="utf-8") == "server:\n  port: 9000\n"


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
        assert settings.embedding_batch_max_texts == 32
        assert settings.rerank_batch_max_documents == 4
        assert settings.rerank_batch_max_tokens == 2048
        assert settings.inference_max_concurrency_per_model == 1
        assert settings.inference_max_queue_per_model == 8
        assert settings.inference_queue_timeout_seconds == 180.0
        assert settings.memory_guard_enabled is True
        assert settings.memory_load_guard_enabled is True
        assert settings.memory_load_headroom_fraction == 0.10
        assert settings.memory_min_free_bytes is None
        assert settings.memory_model_size_multiplier == 1.20
        assert settings.memory_generation_guard_enabled is True
        assert settings.memory_generation_kv_bytes_per_token == 0
        assert settings.memory_vlm_image_tokens_per_image == 576
        assert settings.memory_remote_model_estimates_bytes["FLUX.1-schnell"] > 0
        assert settings.retrieval_clear_mlx_cache_after_request is True
        assert settings.generation_worker_isolation_enabled is False
        assert settings.generation_worker_mode == "type"
        assert settings.generation_worker_host == "127.0.0.1"
        assert settings.generation_worker_ready_timeout_seconds == 45.0
        assert settings.generation_worker_shutdown_timeout_seconds == 10.0
        assert settings.generation_worker_idle_timeout_seconds == 1800.0
        assert settings.generation_clear_mlx_cache_after_request is True
        assert settings.generation_kv_bits is None
        assert settings.generation_kv_group_size == 64
        assert settings.generation_quantized_kv_start == 0
        assert settings.generation_prompt_cache_enabled is True
        assert settings.generation_prompt_cache_checkpoint_enabled is False
        assert settings.generation_prompt_cache_max_entries == 4
        assert settings.generation_prompt_cache_min_tokens == 32
        assert settings.service_auto_start is True
        assert settings.service_keep_alive is True
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
