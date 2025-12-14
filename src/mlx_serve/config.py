"""Configuration management for mlx-serve."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlx_serve.core.config_loader import get_config_values


class YamlConfigSettingsSource:
    """Custom settings source for YAML config file."""

    def __init__(self, settings_cls):
        self.settings_cls = settings_cls

    def __call__(self):
        """Load settings from YAML config file."""
        return get_config_values()


class Settings(BaseSettings):
    """Application settings with YAML file, environment variable, and CLI support.

    Configuration priority (highest to lowest):
    1. CLI options (passed directly to functions)
    2. Environment variables (MLX_SERVE_*)
    3. YAML config file (~/.mlx-serve/config.yaml)
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="MLX_SERVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Model storage
    models_dir: Path = Field(
        default=Path.home() / ".mlx-serve" / "models",
        description="Directory for storing MLX models",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level",
    )
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="Log format (text or json)",
    )

    # Cache settings
    cache_max_embedding_models: int = Field(
        default=3,
        description="Maximum number of embedding models to keep in cache (LRU)",
    )
    cache_max_reranker_models: int = Field(
        default=2,
        description="Maximum number of reranker models to keep in cache (LRU)",
    )
    cache_ttl_seconds: int = Field(
        default=1800,
        description="Time-to-live for cached models in seconds (30 minutes default)",
    )

    # Preload settings
    preload_models: list[str] = Field(
        default_factory=list,
        description="List of model names to preload at startup",
    )

    # Batch processing settings
    batch_max_size: int = Field(
        default=32,
        description="Maximum batch size for continuous batching",
    )
    batch_max_wait_ms: int = Field(
        default=50,
        description="Maximum wait time in milliseconds for batch collection",
    )

    # Metrics settings
    metrics_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint",
    )
    metrics_port: int = Field(
        default=9090,
        description="Port for Prometheus metrics endpoint",
    )

    # Auto-download settings
    auto_download: bool = Field(
        default=False,
        description="Automatically download models if not found",
    )
    auto_download_timeout: int = Field(
        default=300,
        description="Timeout in seconds for auto-download",
    )

    @field_validator("preload_models", mode="before")
    @classmethod
    def parse_preload_models(cls, v):
        """Parse preload_models from comma-separated string or list."""
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    @field_validator("models_dir", mode="before")
    @classmethod
    def expand_models_dir(cls, v):
        """Expand ~ in models_dir path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customize settings sources to include YAML config.

        Order (highest priority first):
        1. init_settings (programmatic)
        2. env_settings (environment variables)
        3. dotenv_settings (.env file)
        4. yaml_config_settings_source (YAML config)
        5. file_secret_settings
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


# Global settings instance
settings = Settings()
