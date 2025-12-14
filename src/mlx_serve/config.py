"""Configuration management for mlx-serve."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

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

    @field_validator("preload_models", mode="before")
    @classmethod
    def parse_preload_models(cls, v):
        """Parse preload_models from comma-separated string or list."""
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    def ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
