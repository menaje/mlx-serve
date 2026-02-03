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

ModelType = Literal["embedding", "reranker", "llm", "vlm", "tts", "stt", "image_gen"]

# Model aliases for short names
MODEL_ALIASES: dict[str, tuple[str, ModelType]] = {
    # Embedding models
    "qwen-embedding": ("Qwen/Qwen3-Embedding-0.6B", "embedding"),
    "qwen-embedding-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "embedding"),
    "bge-small": ("BAAI/bge-small-en-v1.5", "embedding"),
    "bge-base": ("BAAI/bge-base-en-v1.5", "embedding"),
    "bge-large": ("BAAI/bge-large-en-v1.5", "embedding"),
    # Reranker models
    "qwen-reranker": ("Qwen/Qwen3-Reranker-0.6B", "reranker"),
    "qwen-reranker-0.6b": ("Qwen/Qwen3-Reranker-0.6B", "reranker"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "reranker"),
    "bge-reranker-large": ("BAAI/bge-reranker-large", "reranker"),
    # LLM models
    "llama-3.2-1b": ("mlx-community/Llama-3.2-1B-Instruct-4bit", "llm"),
    "llama-3.2-3b": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "llm"),
    "qwen2.5-3b": ("mlx-community/Qwen2.5-3B-Instruct-4bit", "llm"),
    "qwen2.5-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "llm"),
    "mistral-7b": ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "llm"),
    # VLM models
    "qwen2-vl-2b": ("mlx-community/Qwen2-VL-2B-Instruct-4bit", "vlm"),
    "qwen2-vl-7b": ("mlx-community/Qwen2-VL-7B-Instruct-4bit", "vlm"),
    "llava-1.5-7b": ("mlx-community/llava-1.5-7b-4bit", "vlm"),
    # TTS models
    "kokoro": ("prince-canuma/Kokoro-82M", "tts"),
    "kokoro-82m": ("prince-canuma/Kokoro-82M", "tts"),
    # STT models
    "whisper-large-v3-turbo": ("mlx-community/whisper-large-v3-turbo", "stt"),
    "whisper-small": ("mlx-community/whisper-small-mlx", "stt"),
    # Image generation models
    "flux-schnell": ("black-forest-labs/FLUX.1-schnell", "image_gen"),
    "flux-dev": ("black-forest-labs/FLUX.1-dev", "image_gen"),
}


def resolve_model_alias(model_name: str) -> tuple[str, str, ModelType | None]:
    """
    Resolve model alias to HuggingFace repo and model type.

    Args:
        model_name: Model name or alias.

    Returns:
        Tuple of (resolved_name, hf_repo, model_type).
        If not an alias, returns (model_name, model_name, None).
    """
    lower_name = model_name.lower()
    if lower_name in MODEL_ALIASES:
        hf_repo, model_type = MODEL_ALIASES[lower_name]
        resolved_name = hf_repo.split("/")[-1]
        return resolved_name, hf_repo, model_type
    return model_name, model_name, None


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
        self._llm_cache = TTLLRUCache(
            maxsize=settings.cache_max_llm_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._vlm_cache = TTLLRUCache(
            maxsize=settings.cache_max_vlm_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._tts_cache = TTLLRUCache(
            maxsize=settings.cache_max_tts_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._stt_cache = TTLLRUCache(
            maxsize=settings.cache_max_stt_models,
            ttl=settings.cache_ttl_seconds,
        )
        self._image_gen_cache = TTLLRUCache(
            maxsize=settings.cache_max_image_gen_models,
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
                caches = [
                    ("embedding", self._embedding_cache),
                    ("reranker", self._reranker_cache),
                    ("llm", self._llm_cache),
                    ("vlm", self._vlm_cache),
                    ("tts", self._tts_cache),
                    ("stt", self._stt_cache),
                    ("image_gen", self._image_gen_cache),
                ]
                for cache_name, cache in caches:
                    expired = cache.cleanup_expired()
                    if expired:
                        logger.info(f"Cleaned up expired {cache_name} models: {expired}")

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

        # Remove from all caches
        self._embedding_cache.remove(model_name)
        self._reranker_cache.remove(model_name)
        self._llm_cache.remove(model_name)
        self._vlm_cache.remove(model_name)
        self._tts_cache.remove(model_name)
        self._stt_cache.remove(model_name)
        self._image_gen_cache.remove(model_name)

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
        """Get or load an embedding model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        # Resolve alias
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._embedding_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Embedding model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        # Try auto-download if enabled and model not found
        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "embedding", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading embedding model: {resolved_name}")

        # Import here to avoid loading MLX at module import time
        from mlx_embeddings import load

        model, tokenizer = load(str(model_dir))
        self._embedding_cache.set(resolved_name, (model, tokenizer))
        return model, tokenizer

    def get_reranker_model(self, model_name: str) -> Any:
        """Get or load a reranker model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        # Resolve alias
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._reranker_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Reranker model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        # Try auto-download if enabled and model not found
        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "reranker", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading reranker model: {resolved_name}")

        # Import here to avoid loading MLX at module import time
        from mlx_lm import load

        model, tokenizer = load(str(model_dir))
        self._reranker_cache.set(resolved_name, (model, tokenizer))
        return model, tokenizer

    def get_llm_model(self, model_name: str) -> Any:
        """Get or load an LLM model for text generation.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._llm_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"LLM model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "llm", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading LLM model: {resolved_name}")

        from mlx_lm import load

        model, tokenizer = load(str(model_dir))
        self._llm_cache.set(resolved_name, (model, tokenizer))
        return model, tokenizer

    def get_vlm_model(self, model_name: str) -> Any:
        """Get or load a Vision-Language model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._vlm_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"VLM model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "vlm", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading VLM model: {resolved_name}")

        from mlx_vlm import load

        model, processor = load(str(model_dir))
        self._vlm_cache.set(resolved_name, (model, processor))
        return model, processor

    def get_tts_model(self, model_name: str) -> Any:
        """Get or load a TTS (Text-to-Speech) model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._tts_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"TTS model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "tts", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading TTS model: {resolved_name}")

        from mlx_audio.tts import load

        model = load(str(model_dir))
        self._tts_cache.set(resolved_name, model)
        return model

    def get_stt_model(self, model_name: str) -> Any:
        """Get or load an STT (Speech-to-Text) model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._stt_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"STT model cache hit: {resolved_name}")
            return cached

        model_dir = self._get_model_dir(resolved_name)

        if not model_dir.exists():
            if settings.auto_download:
                logger.info(f"Model '{resolved_name}' not found, attempting auto-download...")
                success = self._auto_download_model(hf_repo, "stt", resolved_name)
                if not success:
                    raise ValueError(f"Model '{model_name}' not found and auto-download failed")
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    "Enable auto_download or use 'mlx-serve pull'"
                )

        logger.info(f"Loading STT model: {resolved_name}")

        from mlx_audio.stt import load

        model = load(str(model_dir))
        self._stt_cache.set(resolved_name, model)
        return model

    def get_image_gen_model(self, model_name: str) -> Any:
        """Get or load an image generation model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        cached = self._image_gen_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Image gen model cache hit: {resolved_name}")
            return cached

        # For FLUX models, we use mflux which handles model loading differently
        logger.info(f"Loading image generation model: {resolved_name}")

        from mflux import Flux1

        # Determine model variant from alias
        if "schnell" in resolved_name.lower() or "schnell" in hf_repo.lower():
            model = Flux1.from_alias("flux1-schnell")
        else:
            model = Flux1.from_alias("flux1-dev")

        self._image_gen_cache.set(resolved_name, model)
        return model

    def _auto_download_model(
        self,
        hf_repo: str,
        model_type: ModelType,
        model_name: str | None = None,
    ) -> bool:
        """
        Automatically download a model from HuggingFace.

        Args:
            hf_repo: HuggingFace repository ID.
            model_type: Type of model (embedding or reranker).
            model_name: Optional custom name for the model.

        Returns:
            True if download succeeded, False otherwise.
        """
        import asyncio
        import concurrent.futures

        if model_name is None:
            model_name = hf_repo.split("/")[-1]

        timeout = settings.auto_download_timeout

        try:
            # Run async download synchronously with timeout
            async def _download():
                async for status in self.pull_model(hf_repo, model_type, model_name):
                    if status["status"] == "error":
                        err_msg = status.get('message', 'Unknown error')
                        logger.error(f"Auto-download failed: {err_msg}")
                        return False
                    elif status["status"] == "success":
                        return True
                return False

            # Use ThreadPoolExecutor to run with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _download())
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Auto-download timed out after {timeout}s for {hf_repo}")
                    return False

        except Exception as e:
            logger.error(f"Auto-download failed for {hf_repo}: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "embedding_models": {
                "count": len(self._embedding_cache),
                "max_size": settings.cache_max_embedding_models,
                "models": self._embedding_cache.keys(),
            },
            "llm_models": {
                "count": len(self._llm_cache),
                "max_size": settings.cache_max_llm_models,
                "models": self._llm_cache.keys(),
            },
            "vlm_models": {
                "count": len(self._vlm_cache),
                "max_size": settings.cache_max_vlm_models,
                "models": self._vlm_cache.keys(),
            },
            "tts_models": {
                "count": len(self._tts_cache),
                "max_size": settings.cache_max_tts_models,
                "models": self._tts_cache.keys(),
            },
            "stt_models": {
                "count": len(self._stt_cache),
                "max_size": settings.cache_max_stt_models,
                "models": self._stt_cache.keys(),
            },
            "image_gen_models": {
                "count": len(self._image_gen_cache),
                "max_size": settings.cache_max_image_gen_models,
                "models": self._image_gen_cache.keys(),
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
            model_type = model_info.model_type
            loaders = {
                "embedding": self.get_embedding_model,
                "reranker": self.get_reranker_model,
                "llm": self.get_llm_model,
                "vlm": self.get_vlm_model,
                "tts": self.get_tts_model,
                "stt": self.get_stt_model,
                "image_gen": self.get_image_gen_model,
            }
            if model_type in loaders:
                loaders[model_type](model_name)
                logger.info(f"Preloaded {model_type} model: {model_name}")
            else:
                logger.warning(f"Unknown model type '{model_type}' for model '{model_name}'")
                return False
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
