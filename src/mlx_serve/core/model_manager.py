"""Model manager for loading, caching, and managing MLX models."""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from huggingface_hub import snapshot_download

from mlx_serve.config import settings

logger = logging.getLogger(__name__)

ModelType = Literal["embedding", "reranker"]


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    model_type: ModelType
    path: Path
    size: int = 0
    modified_at: str = ""
    hf_repo: str = ""


@dataclass
class ModelCache:
    """Cache for loaded models."""

    embeddings: dict[str, Any] = field(default_factory=dict)
    rerankers: dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Singleton manager for MLX models."""

    _instance: "ModelManager | None" = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._cache = ModelCache()
        self._metadata_path = settings.models_dir.parent / "metadata.json"
        self._metadata: dict[str, dict] = {}

        settings.ensure_dirs()
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self._metadata_path.exists():
            self._metadata = json.loads(self._metadata_path.read_text())
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(json.dumps(self._metadata, indent=2))

    def _get_model_dir(self, model_name: str) -> Path:
        """Get the directory path for a model."""
        safe_name = model_name.replace("/", "--")
        return settings.models_dir / safe_name

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        for file in path.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total

    def list_models(self) -> list[ModelInfo]:
        """List all installed models."""
        models = []
        for name, meta in self._metadata.items():
            model_dir = self._get_model_dir(name)
            if model_dir.exists():
                models.append(
                    ModelInfo(
                        name=name,
                        model_type=meta.get("type", "embedding"),
                        path=model_dir,
                        size=meta.get("size", 0),
                        modified_at=meta.get("modified_at", ""),
                        hf_repo=meta.get("hf_repo", ""),
                    )
                )
        return models

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a specific model."""
        if model_name not in self._metadata:
            return None

        meta = self._metadata[model_name]
        model_dir = self._get_model_dir(model_name)

        if not model_dir.exists():
            return None

        return ModelInfo(
            name=model_name,
            model_type=meta.get("type", "embedding"),
            path=model_dir,
            size=meta.get("size", 0),
            modified_at=meta.get("modified_at", ""),
            hf_repo=meta.get("hf_repo", ""),
        )

    def is_model_installed(self, model_name: str) -> bool:
        """Check if a model is installed."""
        return model_name in self._metadata and self._get_model_dir(model_name).exists()

    async def pull_model(
        self,
        hf_repo: str,
        model_type: ModelType,
        model_name: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Download and convert a model from Hugging Face.

        Yields status updates during the process.
        """
        if model_name is None:
            model_name = hf_repo.split("/")[-1]

        model_dir = self._get_model_dir(model_name)

        try:
            yield {"status": "downloading", "name": model_name}

            # Download from Hugging Face
            snapshot_download(
                repo_id=hf_repo,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )

            yield {"status": "converting", "name": model_name}

            # For MLX models, conversion is handled by the respective libraries
            # mlx-embeddings and mlx-lm handle conversion on first load

            # Update metadata
            self._metadata[model_name] = {
                "type": model_type,
                "hf_repo": hf_repo,
                "size": self._get_dir_size(model_dir),
                "modified_at": datetime.now().isoformat(),
            }
            self._save_metadata()

            yield {"status": "success", "name": model_name}

        except Exception as e:
            logger.error(f"Failed to pull model {hf_repo}: {e}")
            yield {"status": "error", "message": str(e)}

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from disk."""
        model_dir = self._get_model_dir(model_name)

        # Remove from cache
        self._cache.embeddings.pop(model_name, None)
        self._cache.rerankers.pop(model_name, None)

        # Remove from metadata
        if model_name in self._metadata:
            del self._metadata[model_name]
            self._save_metadata()

        # Remove files
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True

        return False

    def get_embedding_model(self, model_name: str) -> Any:
        """Get or load an embedding model."""
        if model_name in self._cache.embeddings:
            return self._cache.embeddings[model_name]

        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            raise ValueError(f"Model '{model_name}' not found")

        # Import here to avoid loading MLX at module import time
        from mlx_embeddings import load

        model, tokenizer = load(str(model_dir))
        self._cache.embeddings[model_name] = (model, tokenizer)
        return model, tokenizer

    def get_reranker_model(self, model_name: str) -> Any:
        """Get or load a reranker model."""
        if model_name in self._cache.rerankers:
            return self._cache.rerankers[model_name]

        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            raise ValueError(f"Model '{model_name}' not found")

        # Import here to avoid loading MLX at module import time
        from mlx_lm import load

        model, tokenizer = load(str(model_dir))
        self._cache.rerankers[model_name] = (model, tokenizer)
        return model, tokenizer


# Global instance
model_manager = ModelManager()
