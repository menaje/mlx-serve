"""Model manager for loading, caching, and managing MLX models."""

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from cachetools import LRUCache
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


class TTLLRUCache:
    """LRU cache with TTL (Time-To-Live) support."""

    def __init__(self, maxsize: int, ttl: int):
        """Initialize cache with max size and TTL in seconds."""
        self._cache: LRUCache = LRUCache(maxsize=maxsize)
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get item from cache, returning None if expired or not found."""
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                return None

            # Update timestamp on access (refresh TTL)
            self._timestamps[key] = time.time()
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            return self._remove(key)

    def _remove(self, key: str) -> bool:
        """Internal remove without lock."""
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            return True
        return False

    def _is_expired(self, key: str) -> bool:
        """Check if an item has expired."""
        if key not in self._timestamps:
            return True
        return (time.time() - self._timestamps[key]) > self._ttl

    def cleanup_expired(self) -> list[str]:
        """Remove all expired items and return their keys."""
        with self._lock:
            expired = [k for k in list(self._cache.keys()) if self._is_expired(k)]
            for key in expired:
                self._remove(key)
            return expired

    def keys(self) -> list[str]:
        """Return list of keys in cache."""
        with self._lock:
            return list(self._cache.keys())

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache and not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)


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
        self._embedding_cache = TTLLRUCache(
            maxsize=settings.cache_max_embedding_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._reranker_cache = TTLLRUCache(
            maxsize=settings.cache_max_reranker_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._metadata_path = settings.models_dir.parent / "metadata.json"
        self._metadata: dict[str, dict] = {}

        settings.ensure_dirs()
        self._load_metadata()

        # Start background cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start background thread for TTL cleanup."""

        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                expired_embeddings = self._embedding_cache.cleanup_expired()
                expired_rerankers = self._reranker_cache.cleanup_expired()
                if expired_embeddings:
                    logger.info(f"Cleaned up expired embedding models: {expired_embeddings}")
                if expired_rerankers:
                    logger.info(f"Cleaned up expired reranker models: {expired_rerankers}")

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

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
        self._embedding_cache.remove(model_name)
        self._reranker_cache.remove(model_name)

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
        cached = self._embedding_cache.get(model_name)
        if cached is not None:
            logger.debug(f"Embedding model cache hit: {model_name}")
            return cached

        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            raise ValueError(f"Model '{model_name}' not found")

        logger.info(f"Loading embedding model: {model_name}")

        # Import here to avoid loading MLX at module import time
        from mlx_embeddings import load

        model, tokenizer = load(str(model_dir))
        self._embedding_cache.set(model_name, (model, tokenizer))
        return model, tokenizer

    def get_reranker_model(self, model_name: str) -> Any:
        """Get or load a reranker model."""
        cached = self._reranker_cache.get(model_name)
        if cached is not None:
            logger.debug(f"Reranker model cache hit: {model_name}")
            return cached

        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            raise ValueError(f"Model '{model_name}' not found")

        logger.info(f"Loading reranker model: {model_name}")

        # Import here to avoid loading MLX at module import time
        from mlx_lm import load

        model, tokenizer = load(str(model_dir))
        self._reranker_cache.set(model_name, (model, tokenizer))
        return model, tokenizer

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "embedding_models": {
                "count": len(self._embedding_cache),
                "max_size": settings.cache_max_embedding_models,
                "models": self._embedding_cache.keys(),
            },
            "reranker_models": {
                "count": len(self._reranker_cache),
                "max_size": settings.cache_max_reranker_models,
                "models": self._reranker_cache.keys(),
            },
            "ttl_seconds": settings.cache_ttl_seconds,
        }

    def preload_model(self, model_name: str) -> bool:
        """Preload a model into cache.

        Args:
            model_name: Name of the model to preload.

        Returns:
            True if model was successfully loaded, False otherwise.
        """
        if not self.is_model_installed(model_name):
            logger.warning(f"Model '{model_name}' not installed, cannot preload")
            return False

        model_info = self.get_model_info(model_name)
        if model_info is None:
            logger.warning(f"Model '{model_name}' info not found, cannot preload")
            return False

        try:
            if model_info.model_type == "embedding":
                self.get_embedding_model(model_name)
                logger.info(f"Preloaded embedding model: {model_name}")
            elif model_info.model_type == "reranker":
                self.get_reranker_model(model_name)
                logger.info(f"Preloaded reranker model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model '{model_name}': {e}")
            return False

    def preload_models(self, model_names: list[str] | None = None) -> dict[str, bool]:
        """Preload multiple models.

        Args:
            model_names: List of model names to preload. If None, uses settings.preload_models.

        Returns:
            Dict mapping model names to success status.
        """
        if model_names is None:
            model_names = settings.preload_models

        if not model_names:
            return {}

        results = {}
        for model_name in model_names:
            results[model_name] = self.preload_model(model_name)
        return results


# Global instance
model_manager = ModelManager()
