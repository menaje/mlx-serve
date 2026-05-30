"""Model manager for loading, caching, and managing MLX models."""

import gc
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from cachetools import LRUCache
from huggingface_hub import snapshot_download

from mlx_serve.config import settings
from mlx_serve.core.mlx_memory import clear_mlx_cache, get_mlx_memory_snapshot
from mlx_serve.core.model_memory import (
    GenerationMemoryError,
    ModelLoadMemoryError,
    check_generation_memory,
    check_model_load_memory,
    estimate_generation_kv_cache_bytes,
    remote_model_estimate_bytes,
)
from mlx_serve.core.model_memory import (
    estimate_vlm_image_tokens as estimate_vlm_image_tokens_for_model_dir,
)
from mlx_serve.core.system_guard import collect_memory_snapshot

logger = logging.getLogger(__name__)

ModelType = Literal["embedding", "reranker", "llm", "vlm", "tts", "stt", "image_gen"]
ModelUnloadHook = Callable[[ModelType, str, str], None]

_model_unload_hooks: list[ModelUnloadHook] = []
_model_unload_hooks_lock = threading.Lock()


def register_model_unload_hook(hook: ModelUnloadHook) -> None:
    """Register a callback fired when a cached model is removed."""
    with _model_unload_hooks_lock:
        if hook not in _model_unload_hooks:
            _model_unload_hooks.append(hook)

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


class ModelInUseError(RuntimeError):
    """Raised when a loaded model cannot be unloaded because it is active."""


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

    def set(self, key: str, value: Any) -> list[str]:
        """Set item in cache and return keys evicted by maxsize pressure."""
        with self._lock:
            before_keys = list(self._cache.keys())
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._prune_timestamps()
            after_keys = set(self._cache.keys())
            return [evicted_key for evicted_key in before_keys if evicted_key not in after_keys]

    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            return self._remove(key)

    def pop(self, key: str) -> Any | None:
        """Remove and return an item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            value = self._cache[key]
            self._remove(key)
            return value

    def _remove(self, key: str) -> bool:
        """Internal remove without lock."""
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            return True
        return False

    def _prune_timestamps(self) -> None:
        """Remove timestamps for entries evicted by the underlying LRU cache."""
        for key in list(self._timestamps):
            if key not in self._cache:
                self._timestamps.pop(key, None)

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

    def items_by_last_used(self) -> list[tuple[str, float, Any]]:
        """Return cache items ordered from least to most recently used."""
        with self._lock:
            expired = [k for k in list(self._cache.keys()) if self._is_expired(k)]
            for key in expired:
                self._remove(key)
            items = [
                (key, self._timestamps.get(key, 0.0), self._cache[key])
                for key in self._cache.keys()
            ]
        return sorted(items, key=lambda item: item[1])

    def last_used(self, key: str) -> float | None:
        """Return the last-used timestamp for a key."""
        with self._lock:
            return self._timestamps.get(key)

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
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        with self._lock:
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
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_guard = threading.Lock()
        self._model_estimated_weight_bytes: dict[tuple[ModelType, str], int] = {}

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
                    self._cleanup_expired_cache(
                        cache_name,
                        cache,
                        f"expired {cache_name} cleanup",
                    )

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self._metadata_path.exists():
            try:
                self._metadata = json.loads(self._metadata_path.read_text())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse metadata file, resetting: {e}")
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(json.dumps(self._metadata, indent=2))

    def _get_load_lock(self, key: str) -> threading.Lock:
        """Get a per-model lock for cache misses and model loading."""
        with self._load_locks_guard:
            lock = self._load_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[key] = lock
            return lock

    def _get_model_dir(self, model_name: str) -> Path:
        """Get the directory path for a model."""
        safe_name = model_name.replace("/", "--")
        return settings.models_dir / safe_name

    def _cache_map(self) -> dict[ModelType, TTLLRUCache]:
        """Return caches keyed by model type."""
        return {
            "embedding": self._embedding_cache,
            "reranker": self._reranker_cache,
            "llm": self._llm_cache,
            "vlm": self._vlm_cache,
            "tts": self._tts_cache,
            "stt": self._stt_cache,
            "image_gen": self._image_gen_cache,
        }

    def _active_inference_keys(self) -> set[str]:
        """Return currently active inference keys, best-effort."""
        try:
            from mlx_serve.core.inference_control import inference_controller

            snapshot = inference_controller.snapshot()
        except Exception:
            return set()
        return {
            key
            for key, state in snapshot.items()
            if int(state.get("active", 0)) > 0
        }

    def _model_inference_key(self, model_type: ModelType, model_name: str) -> str:
        """Build the cache/inference key form without importing inference_control."""
        return f"{model_type}:{model_name}"

    def _estimate_key(self, model_type: ModelType, model_name: str) -> tuple[ModelType, str]:
        """Build a key for cached model memory estimates."""
        return (model_type, model_name)

    def _forget_model_estimate(self, model_type: ModelType, model_name: str) -> None:
        """Remove a cached model memory estimate."""
        self._model_estimated_weight_bytes.pop(self._estimate_key(model_type, model_name), None)

    def _notify_cached_model_removed(
        self,
        model_type: ModelType,
        model_name: str,
        reason: str,
        *,
        record_eviction: bool = False,
    ) -> None:
        """Run cleanup side effects for a model removed from a cache."""
        self._forget_model_estimate(model_type, model_name)
        if record_eviction:
            try:
                from mlx_serve.core.metrics import record_cache_eviction

                record_cache_eviction(model_type)
            except Exception:
                pass

        with _model_unload_hooks_lock:
            hooks = list(_model_unload_hooks)

        for hook in hooks:
            try:
                hook(model_type, model_name, reason)
            except Exception:
                logger.warning(
                    "Model unload hook failed for %s model '%s' (%s)",
                    model_type,
                    model_name,
                    reason,
                    exc_info=True,
                )

    def _cache_loaded_model(
        self,
        model_type: ModelType,
        model_name: str,
        cache: TTLLRUCache,
        value: Any,
    ) -> None:
        """Store a loaded model and clean up entries evicted by LRU pressure."""
        evicted = cache.set(model_name, value)
        if not evicted:
            return

        for evicted_name in evicted:
            logger.info(
                "Evicted LRU %s model '%s' while loading '%s'",
                model_type,
                evicted_name,
                model_name,
            )
            self._notify_cached_model_removed(
                model_type,
                evicted_name,
                f"lru {model_type} cache",
                record_eviction=True,
            )
        gc.collect()
        clear_mlx_cache(log=logger, reason=f"lru {model_type} cache")

    def _cleanup_expired_cache(
        self,
        model_type: ModelType,
        cache: TTLLRUCache,
        reason: str,
    ) -> list[str]:
        """Remove expired models with the same cleanup hooks as explicit unload."""
        expired = cache.cleanup_expired()
        if not expired:
            return []

        for model_name in expired:
            self._notify_cached_model_removed(
                model_type,
                model_name,
                reason,
                record_eviction=True,
            )
        logger.info("Cleaned up expired %s models: %s", model_type, expired)
        gc.collect()
        clear_mlx_cache(log=logger, reason=reason)
        return expired

    def _cached_estimated_weight_bytes(self, model_type: ModelType, model_name: str) -> int:
        """Return a stored estimate, falling back to install metadata."""
        estimate = self._model_estimated_weight_bytes.get(
            self._estimate_key(model_type, model_name)
        )
        if estimate is not None:
            return estimate
        metadata_size = self._metadata.get(model_name, {}).get("size")
        return int(metadata_size or 0)

    def _idle_eviction_candidates(
        self,
        target_keys: set[str] | None = None,
    ) -> list[tuple[int, float, ModelType, str, TTLLRUCache]]:
        """Return idle cached models ordered by likely memory benefit."""
        active_keys = self._active_inference_keys()
        target_keys = target_keys or set()
        candidates: list[tuple[int, float, ModelType, str, TTLLRUCache]] = []
        for candidate_type, cache in self._cache_map().items():
            for key, last_used, _value in cache.items_by_last_used():
                inference_key = self._model_inference_key(candidate_type, key)
                if inference_key in target_keys or inference_key in active_keys:
                    continue
                estimated_bytes = self._cached_estimated_weight_bytes(candidate_type, key)
                candidates.append((estimated_bytes, last_used, candidate_type, key, cache))
        return sorted(candidates, key=lambda item: (-item[0], item[1]))

    def _evict_idle_model(
        self,
        model_type: ModelType,
        model_name: str,
        cache: TTLLRUCache,
        reason: str,
    ) -> bool:
        """Evict one idle model and clear runtime cache."""
        if not cache.remove(model_name):
            return False
        self._notify_cached_model_removed(
            model_type,
            model_name,
            reason,
            record_eviction=True,
        )
        logger.info("Evicted idle %s model '%s' for %s", model_type, model_name, reason)
        gc.collect()
        clear_mlx_cache(log=logger, reason=f"evict idle {model_type} {model_name}")
        return True

    def _clear_prompt_cache_for_pressure(self, reason: str) -> bool:
        """Clear prompt KV cache before evicting loaded models under pressure."""
        try:
            from mlx_serve.core.prompt_cache import prompt_cache_store

            removed = prompt_cache_store.clear()
        except Exception:
            logger.debug("Failed to clear prompt cache for %s", reason, exc_info=True)
            return False
        if removed <= 0:
            return False
        logger.info(
            "Cleared %s prompt cache entr%s for %s",
            removed,
            "y" if removed == 1 else "ies",
            reason,
        )
        gc.collect()
        clear_mlx_cache(log=logger, reason=f"clear prompt cache for {reason}")
        return True

    def _check_and_store_load_estimate(
        self,
        model_type: ModelType,
        model_name: str,
        model_dir: Path | None = None,
        estimated_weight_bytes: int | None = None,
        extra_required_bytes: int = 0,
    ) -> None:
        """Run load admission and remember the resulting estimate."""
        estimate = check_model_load_memory(
            model_name,
            model_type,
            model_dir,
            estimated_weight_bytes=estimated_weight_bytes,
            extra_required_bytes=extra_required_bytes,
        )
        self._model_estimated_weight_bytes[self._estimate_key(model_type, model_name)] = (
            estimate.estimated_weight_bytes
        )

    def _ensure_load_memory_available(
        self,
        model_type: ModelType,
        model_name: str,
        model_dir: Path | None = None,
        estimated_weight_bytes: int | None = None,
        extra_required_bytes: int = 0,
    ) -> None:
        """Check load memory and evict idle models if needed."""
        try:
            self._check_and_store_load_estimate(
                model_type,
                model_name,
                model_dir,
                estimated_weight_bytes,
                extra_required_bytes,
            )
            return
        except ModelLoadMemoryError as first_error:
            last_error = first_error

        if self._clear_prompt_cache_for_pressure(f"loading '{model_name}'"):
            try:
                self._check_and_store_load_estimate(
                    model_type,
                    model_name,
                    model_dir,
                    estimated_weight_bytes,
                    extra_required_bytes,
                )
                return
            except ModelLoadMemoryError as exc:
                last_error = exc

        target_key = self._model_inference_key(model_type, model_name)
        for _estimated_bytes, _last_used, candidate_type, key, cache in (
            self._idle_eviction_candidates({target_key})
        ):
            if not self._evict_idle_model(
                candidate_type,
                key,
                cache,
                reason=f"loading '{model_name}'",
            ):
                continue

            try:
                self._check_and_store_load_estimate(
                    model_type,
                    model_name,
                    model_dir,
                    estimated_weight_bytes,
                    extra_required_bytes,
                )
                return
            except ModelLoadMemoryError as exc:
                last_error = exc

        try:
            from mlx_serve.core.metrics import record_model_load_memory_rejection

            record_model_load_memory_rejection(
                model_type,
                last_error.estimate.required_bytes,
            )
        except Exception:
            logger.debug("Failed to record model load memory rejection", exc_info=True)
        raise last_error

    def ensure_load_and_generation_memory_available(
        self,
        model_type: Literal["llm", "vlm"],
        model_name: str,
        prompt_tokens: int,
        max_tokens: int,
        image_tokens: int = 0,
    ) -> None:
        """Check estimated model weights plus request KV before loading a model."""
        resolved_name, _, _ = resolve_model_alias(model_name)
        model_dir = self._get_model_dir(resolved_name)
        model_dir_arg = model_dir if model_dir.exists() else None
        kv_cache_bytes = estimate_generation_kv_cache_bytes(
            model_dir_arg,
            max(0, int(prompt_tokens)) + max(0, int(max_tokens)) + max(0, int(image_tokens)),
        )
        self._ensure_load_memory_available(
            model_type,
            resolved_name,
            model_dir_arg,
            extra_required_bytes=kv_cache_bytes,
        )

    def check_generation_memory_available(
        self,
        model_type: Literal["llm", "vlm"],
        model_name: str,
        prompt_tokens: int,
        max_tokens: int,
        image_tokens: int = 0,
    ) -> None:
        """Check request-scoped generation memory before allocating KV cache."""
        resolved_name, _, _ = resolve_model_alias(model_name)
        model_dir = self._get_model_dir(resolved_name)
        model_dir_arg = model_dir if model_dir.exists() else None

        try:
            check_generation_memory(
                resolved_name,
                model_type,
                prompt_tokens,
                max_tokens,
                image_tokens=image_tokens,
                model_dir=model_dir_arg,
            )
            return
        except GenerationMemoryError as first_error:
            last_error = first_error

        if self._clear_prompt_cache_for_pressure(f"generation on '{model_name}'"):
            try:
                check_generation_memory(
                    resolved_name,
                    model_type,
                    prompt_tokens,
                    max_tokens,
                    image_tokens=image_tokens,
                    model_dir=model_dir_arg,
                )
                return
            except GenerationMemoryError as exc:
                last_error = exc

        target_keys = {
            self._model_inference_key(model_type, resolved_name),
            self._model_inference_key(model_type, model_name),
        }
        for _estimated_bytes, _last_used, candidate_type, key, cache in (
            self._idle_eviction_candidates(target_keys)
        ):
            if not self._evict_idle_model(
                candidate_type,
                key,
                cache,
                reason=f"generation on '{model_name}'",
            ):
                continue
            try:
                check_generation_memory(
                    resolved_name,
                    model_type,
                    prompt_tokens,
                    max_tokens,
                    image_tokens=image_tokens,
                    model_dir=model_dir_arg,
                )
                return
            except GenerationMemoryError as exc:
                last_error = exc

        raise last_error

    def estimate_prompt_cache_bytes(
        self,
        model_type: Literal["llm", "vlm"],
        model_name: str,
        prompt_tokens: int,
    ) -> int:
        """Estimate bytes held by a reusable prompt KV cache prefix."""
        resolved_name, _, _ = resolve_model_alias(model_name)
        model_dir = self._get_model_dir(resolved_name)
        model_dir_arg = model_dir if model_dir.exists() else None
        return estimate_generation_kv_cache_bytes(model_dir_arg, max(0, int(prompt_tokens)))

    def estimate_vlm_image_tokens(self, model_name: str, image_count: int) -> int:
        """Estimate VLM image tokens for a model using processor/model config."""
        resolved_name, _, _ = resolve_model_alias(model_name)
        model_dir = self._get_model_dir(resolved_name)
        model_dir_arg = model_dir if model_dir.exists() else None
        return estimate_vlm_image_tokens_for_model_dir(model_dir_arg, image_count)

    def _calibrate_loaded_model_estimate(
        self,
        model_type: ModelType,
        model_name: str,
        before_process_rss_bytes: int | None,
        before_mlx_active_bytes: int | None,
    ) -> None:
        """Raise stored estimates from observed load-time memory deltas."""
        after_snapshot = collect_memory_snapshot()
        after_mlx = get_mlx_memory_snapshot()
        observed_candidates: list[int] = []
        if (
            before_process_rss_bytes is not None
            and after_snapshot.process_rss_bytes is not None
        ):
            observed_candidates.append(
                after_snapshot.process_rss_bytes - before_process_rss_bytes
            )
        after_mlx_active = after_mlx.get("active_bytes") if after_mlx.get("available") else None
        if before_mlx_active_bytes is not None and after_mlx_active is not None:
            observed_candidates.append(int(after_mlx_active) - before_mlx_active_bytes)

        observed_bytes = max(observed_candidates, default=0)
        if observed_bytes <= 0:
            return

        key = self._estimate_key(model_type, model_name)
        previous = self._model_estimated_weight_bytes.get(key, 0)
        calibrated = max(previous, observed_bytes)
        self._model_estimated_weight_bytes[key] = calibrated
        logger.info(
            "Calibrated %s model '%s' memory estimate: observed=%s previous=%s stored=%s",
            model_type,
            model_name,
            observed_bytes,
            previous,
            calibrated,
        )

    def _memory_calibration_baseline(self) -> tuple[int | None, int | None]:
        """Capture process and MLX active memory before a model load."""
        before_snapshot = collect_memory_snapshot()
        before_mlx = get_mlx_memory_snapshot()
        before_mlx_active = (
            int(before_mlx["active_bytes"])
            if before_mlx.get("available") and before_mlx.get("active_bytes") is not None
            else None
        )
        return before_snapshot.process_rss_bytes, before_mlx_active

    def capture_memory_calibration_baseline(self) -> tuple[int | None, int | None]:
        """Capture process and MLX active memory before a request allocation."""
        return self._memory_calibration_baseline()

    def calibrate_model_estimate_from_baseline(
        self,
        model_type: ModelType,
        model_name: str,
        baseline: tuple[int | None, int | None],
    ) -> None:
        """Raise stored model estimate from an observed request-time delta."""
        before_process_rss, before_mlx_active = baseline
        self._calibrate_loaded_model_estimate(
            model_type,
            model_name,
            before_process_rss,
            before_mlx_active,
        )

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

    def get_model_type(self, model_name: str) -> ModelType | None:
        """Get the type of an installed model."""
        resolved_name, _, _ = resolve_model_alias(model_name)
        if resolved_name in self._metadata:
            return self._metadata[resolved_name].get("type")
        return None

    async def pull_model(
        self,
        hf_repo: str,
        model_type: ModelType,
        model_name: str | None = None,
        quantize: int | None = None,
        keep_original: bool = False,
    ) -> AsyncIterator[dict]:
        """
        Download and convert a model from Hugging Face to MLX format.

        Args:
            hf_repo: HuggingFace repository ID.
            model_type: Type of model.
            model_name: Optional custom name for the model.
            quantize: Quantization bits (4 or 8). None for no quantization.
            keep_original: If False, delete HF cache after conversion (default: False).

        Yields status updates during the process.
        """
        if model_name is None:
            model_name = hf_repo.split("/")[-1]

        # Add quantization suffix to model name
        if quantize:
            output_name = f"{model_name}-{quantize}bit"
        else:
            output_name = model_name

        model_dir = self._get_model_dir(output_name)

        try:
            yield {"status": "downloading", "name": output_name, "hf_repo": hf_repo}

            # Check if model is already in MLX format (from mlx-community etc)
            is_mlx_model = self._is_mlx_format_repo(hf_repo)

            if is_mlx_model and not quantize:
                # MLX model without quantization: direct download
                snapshot_download(
                    repo_id=hf_repo,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                yield {"status": "converting", "name": output_name, "detail": "Already MLX format"}
            else:
                # Need conversion: download to temp then convert
                yield {"status": "converting", "name": output_name}

                convert_success = await self._convert_model(
                    hf_repo=hf_repo,
                    output_dir=model_dir,
                    model_type=model_type,
                    quantize=quantize,
                )

                if not convert_success:
                    yield {"status": "error", "message": "Conversion failed"}
                    return

                # Clean up HF cache if not keeping original
                if not keep_original:
                    yield {"status": "cleaning", "name": output_name, "detail": "Removing HF cache"}
                    self._cleanup_hf_cache(hf_repo)

            # Update metadata
            metadata_entry = {
                "type": model_type,
                "hf_repo": hf_repo,
                "size": self._get_dir_size(model_dir),
                "modified_at": datetime.now().isoformat(),
            }
            if quantize:
                metadata_entry["quantization"] = {"bits": quantize}

            self._metadata[output_name] = metadata_entry
            self._save_metadata()

            yield {"status": "success", "name": output_name}

        except Exception as e:
            logger.error(f"Failed to pull model {hf_repo}: {e}")
            yield {"status": "error", "message": str(e)}

    def _is_mlx_format_repo(self, hf_repo: str) -> bool:
        """Check if a HuggingFace repo is already in MLX format."""
        mlx_indicators = ["mlx-community", "mlx-", "-mlx", "4bit", "8bit"]
        repo_lower = hf_repo.lower()
        return any(indicator in repo_lower for indicator in mlx_indicators)

    def _cleanup_hf_cache(self, hf_repo: str) -> bool:
        """
        Remove a model from the HuggingFace cache.

        Args:
            hf_repo: HuggingFace repository ID.

        Returns:
            True if cleanup succeeded, False otherwise.
        """
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()

            # Find the repo in cache
            for repo in cache_info.repos:
                if repo.repo_id == hf_repo:
                    # Delete all revisions of this repo
                    delete_strategy = cache_info.delete_revisions(
                        *[rev.commit_hash for rev in repo.revisions]
                    )
                    delete_strategy.execute()
                    # Log cleanup (freed_size_str may not exist in all versions)
                    freed_size = getattr(delete_strategy, 'freed_size_str', 'unknown size')
                    logger.info(f"Cleaned up HF cache for {hf_repo}: freed {freed_size}")
                    return True

            logger.debug(f"Model {hf_repo} not found in HF cache")
            return True  # Not an error if not in cache

        except Exception as e:
            logger.warning(f"Failed to cleanup HF cache for {hf_repo}: {e}")
            return False

    async def _convert_model(
        self,
        hf_repo: str,
        output_dir: Path,
        model_type: ModelType,
        quantize: int | None = None,
    ) -> bool:
        """
        Convert a HuggingFace model to MLX format.

        Args:
            hf_repo: HuggingFace repository ID.
            output_dir: Output directory for converted model.
            model_type: Type of model (determines which converter to use).
            quantize: Quantization bits (4 or 8). None for no quantization.

        Returns:
            True if conversion succeeded, False otherwise.
        """
        import asyncio

        loop = asyncio.get_running_loop()

        def _do_convert():
            try:
                if model_type in ("llm", "reranker"):
                    from mlx_lm import convert

                    convert(
                        hf_repo,
                        mlx_path=str(output_dir),
                        quantize=quantize is not None,
                        q_bits=quantize or 4,
                    )
                elif model_type == "vlm":
                    from mlx_vlm import convert

                    convert(
                        hf_repo,
                        mlx_path=str(output_dir),
                        quantize=quantize is not None,
                        q_bits=quantize or 4,
                    )
                elif model_type == "embedding":
                    try:
                        from mlx_embeddings import convert

                        convert(
                            hf_repo,
                            mlx_path=str(output_dir),
                            quantize=quantize is not None,
                            q_bits=quantize or 4,
                        )
                    except (ImportError, AttributeError):
                        # mlx_embeddings may not have convert, use snapshot_download
                        snapshot_download(
                            repo_id=hf_repo,
                            local_dir=str(output_dir),
                            local_dir_use_symlinks=False,
                        )
                else:
                    # For tts, stt, image_gen: direct download (specialized formats)
                    snapshot_download(
                        repo_id=hf_repo,
                        local_dir=str(output_dir),
                        local_dir_use_symlinks=False,
                    )

                return True
            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                return False

        return await loop.run_in_executor(None, _do_convert)

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from disk."""
        model_dir = self._get_model_dir(model_name)

        # Remove from all caches
        for model_type, cache in self._cache_map().items():
            if cache.remove(model_name):
                self._notify_cached_model_removed(
                    model_type,
                    model_name,
                    f"delete model {model_name}",
                )
            else:
                self._forget_model_estimate(model_type, model_name)

        # Remove from metadata
        if model_name in self._metadata:
            del self._metadata[model_name]
            self._save_metadata()

        # Remove files
        if model_dir.exists():
            shutil.rmtree(model_dir)
            gc.collect()
            clear_mlx_cache(log=logger, reason=f"delete model {model_name}")
            return True

        return False

    def unload_model(
        self,
        model_name: str,
        model_type: ModelType | None = None,
    ) -> list[dict[str, str]]:
        """Unload a cached model without deleting it from disk."""
        resolved_name, _, resolved_type = resolve_model_alias(model_name)
        target_types = [model_type] if model_type else list(self._cache_map().keys())
        if model_type is None and resolved_type is not None:
            target_types = [resolved_type]

        active_keys = self._active_inference_keys()
        unloaded: list[dict[str, str]] = []

        for target_type in target_types:
            cache = self._cache_map()[target_type]
            inference_key = self._model_inference_key(target_type, resolved_name)
            if inference_key in active_keys and resolved_name in cache.keys():
                raise ModelInUseError(
                    f"Model '{resolved_name}' is active and cannot be unloaded"
                )
            if cache.remove(resolved_name):
                self._notify_cached_model_removed(
                    target_type,
                    resolved_name,
                    f"unload model {resolved_name}",
                )
                unloaded.append({"name": resolved_name, "type": target_type})

        if unloaded:
            gc.collect()
            clear_mlx_cache(log=logger, reason=f"unload model {resolved_name}")
            try:
                from mlx_serve.core.prompt_cache import prompt_cache_store

                prompt_cache_store.clear(resolved_name)
            except Exception:
                logger.debug("Failed to clear prompt cache for %s", resolved_name, exc_info=True)

        return unloaded

    def unload_all(
        self,
        model_type: ModelType | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        """Unload cached idle models without deleting them from disk."""
        target_types = [model_type] if model_type else list(self._cache_map().keys())
        active_keys = self._active_inference_keys()
        unloaded: list[dict[str, str]] = []
        skipped_active: list[dict[str, str]] = []

        for target_type in target_types:
            cache = self._cache_map()[target_type]
            for key in cache.keys():
                inference_key = self._model_inference_key(target_type, key)
                if inference_key in active_keys:
                    skipped_active.append({"name": key, "type": target_type})
                    continue
                if cache.remove(key):
                    self._notify_cached_model_removed(
                        target_type,
                        key,
                        "unload all models",
                    )
                    unloaded.append({"name": key, "type": target_type})

        if unloaded:
            gc.collect()
            clear_mlx_cache(log=logger, reason="unload all models")
            try:
                from mlx_serve.core.prompt_cache import prompt_cache_store

                if model_type in ("llm", None):
                    prompt_cache_store.clear()
            except Exception:
                logger.debug("Failed to clear prompt cache during unload all", exc_info=True)

        return {"unloaded": unloaded, "skipped_active": skipped_active}

    def get_embedding_model(self, model_name: str) -> Any:
        """Get or load an embedding model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        # Resolve alias
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache(
            "embedding",
            self._embedding_cache,
            "expired embedding cache access",
        )
        cached = self._embedding_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Embedding model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"embedding:{resolved_name}"):
            cached = self._embedding_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"Embedding model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("embedding", resolved_name, model_dir)
            logger.info(f"Loading embedding model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            # Import here to avoid loading MLX at module import time
            from mlx_embeddings import load

            model, tokenizer = load(str(model_dir))
            self._cache_loaded_model(
                "embedding",
                resolved_name,
                self._embedding_cache,
                (model, tokenizer),
            )
            self._calibrate_loaded_model_estimate(
                "embedding",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model, tokenizer

    def get_reranker_model(self, model_name: str) -> Any:
        """Get or load a reranker model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        # Resolve alias
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache(
            "reranker",
            self._reranker_cache,
            "expired reranker cache access",
        )
        cached = self._reranker_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Reranker model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"reranker:{resolved_name}"):
            cached = self._reranker_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"Reranker model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("reranker", resolved_name, model_dir)
            logger.info(f"Loading reranker model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            # Import here to avoid loading MLX at module import time
            from mlx_lm import load

            model, tokenizer = load(str(model_dir))
            self._cache_loaded_model(
                "reranker",
                resolved_name,
                self._reranker_cache,
                (model, tokenizer),
            )
            self._calibrate_loaded_model_estimate(
                "reranker",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model, tokenizer

    def get_llm_model(self, model_name: str) -> Any:
        """Get or load an LLM model for text generation.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache("llm", self._llm_cache, "expired llm cache access")
        cached = self._llm_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"LLM model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"llm:{resolved_name}"):
            cached = self._llm_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"LLM model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("llm", resolved_name, model_dir)
            logger.info(f"Loading LLM model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            from mlx_lm import load

            model, tokenizer = load(str(model_dir))
            self._cache_loaded_model(
                "llm",
                resolved_name,
                self._llm_cache,
                (model, tokenizer),
            )
            self._calibrate_loaded_model_estimate(
                "llm",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model, tokenizer

    def get_vlm_model(self, model_name: str) -> Any:
        """Get or load a Vision-Language model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache("vlm", self._vlm_cache, "expired vlm cache access")
        cached = self._vlm_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"VLM model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"vlm:{resolved_name}"):
            cached = self._vlm_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"VLM model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("vlm", resolved_name, model_dir)
            logger.info(f"Loading VLM model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            from mlx_vlm import load

            model, processor = load(str(model_dir))
            self._cache_loaded_model(
                "vlm",
                resolved_name,
                self._vlm_cache,
                (model, processor),
            )
            self._calibrate_loaded_model_estimate(
                "vlm",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model, processor

    def get_tts_model(self, model_name: str) -> Any:
        """Get or load a TTS (Text-to-Speech) model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache("tts", self._tts_cache, "expired tts cache access")
        cached = self._tts_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"TTS model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"tts:{resolved_name}"):
            cached = self._tts_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"TTS model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("tts", resolved_name, model_dir)
            logger.info(f"Loading TTS model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            from mlx_audio.tts import load

            model = load(str(model_dir))
            self._cache_loaded_model("tts", resolved_name, self._tts_cache, model)
            self._calibrate_loaded_model_estimate(
                "tts",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model

    def get_stt_model(self, model_name: str) -> Any:
        """Get or load an STT (Speech-to-Text) model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache("stt", self._stt_cache, "expired stt cache access")
        cached = self._stt_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"STT model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"stt:{resolved_name}"):
            cached = self._stt_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"STT model cache hit after wait: {resolved_name}")
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

            self._ensure_load_memory_available("stt", resolved_name, model_dir)
            logger.info(f"Loading STT model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            from mlx_audio.stt import load

            model = load(str(model_dir))
            self._cache_loaded_model("stt", resolved_name, self._stt_cache, model)
            self._calibrate_loaded_model_estimate(
                "stt",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
            return model

    def get_image_gen_model(self, model_name: str) -> Any:
        """Get or load an image generation model.

        If auto_download is enabled and model is not found, attempts to download it.
        """
        resolved_name, hf_repo, _ = resolve_model_alias(model_name)

        self._cleanup_expired_cache(
            "image_gen",
            self._image_gen_cache,
            "expired image_gen cache access",
        )
        cached = self._image_gen_cache.get(resolved_name)
        if cached is not None:
            logger.debug(f"Image gen model cache hit: {resolved_name}")
            return cached

        with self._get_load_lock(f"image_gen:{resolved_name}"):
            cached = self._image_gen_cache.get(resolved_name)
            if cached is not None:
                logger.debug(f"Image gen model cache hit after wait: {resolved_name}")
                return cached

            model_dir = self._get_model_dir(resolved_name)
            if model_dir.exists():
                self._ensure_load_memory_available("image_gen", resolved_name, model_dir)
            else:
                estimated_bytes = remote_model_estimate_bytes(hf_repo, resolved_name)
                if estimated_bytes:
                    self._ensure_load_memory_available(
                        "image_gen",
                        resolved_name,
                            estimated_weight_bytes=estimated_bytes,
                        )

            # For FLUX models, we use mflux which handles model loading differently
            logger.info(f"Loading image generation model: {resolved_name}")
            before_process_rss, before_mlx_active = self._memory_calibration_baseline()

            from mflux import Flux1

            # Determine model variant from alias
            if "schnell" in resolved_name.lower() or "schnell" in hf_repo.lower():
                model = Flux1.from_alias("flux1-schnell")
            else:
                model = Flux1.from_alias("flux1-dev")

            self._cache_loaded_model(
                "image_gen",
                resolved_name,
                self._image_gen_cache,
                model,
            )
            self._calibrate_loaded_model_estimate(
                "image_gen",
                resolved_name,
                before_process_rss,
                before_mlx_active,
            )
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
        def cache_payload(
            model_type: ModelType,
            cache: TTLLRUCache,
            max_size: int,
        ) -> dict:
            models = cache.keys()
            return {
                "count": len(cache),
                "max_size": max_size,
                "models": models,
                "model_details": [
                    {
                        "name": model_name,
                        "type": model_type,
                        "last_used": cache.last_used(model_name),
                        "estimated_weight_bytes": self._cached_estimated_weight_bytes(
                            model_type,
                            model_name,
                        ),
                    }
                    for model_name in models
                ],
            }

        try:
            from mlx_serve.core.prompt_cache import prompt_cache_store

            prompt_cache_stats = prompt_cache_store.stats()
        except Exception:
            logger.debug("Failed to collect prompt cache stats", exc_info=True)
            prompt_cache_stats = {"enabled": False, "count": 0, "entries": []}

        return {
            "embedding_models": cache_payload(
                "embedding",
                self._embedding_cache,
                settings.cache_max_embedding_models,
            ),
            "llm_models": cache_payload("llm", self._llm_cache, settings.cache_max_llm_models),
            "vlm_models": cache_payload("vlm", self._vlm_cache, settings.cache_max_vlm_models),
            "tts_models": cache_payload("tts", self._tts_cache, settings.cache_max_tts_models),
            "stt_models": cache_payload("stt", self._stt_cache, settings.cache_max_stt_models),
            "image_gen_models": cache_payload(
                "image_gen",
                self._image_gen_cache,
                settings.cache_max_image_gen_models,
            ),
            "reranker_models": cache_payload(
                "reranker",
                self._reranker_cache,
                settings.cache_max_reranker_models,
            ),
            "ttl_seconds": settings.cache_ttl_seconds,
            "active_inference": sorted(self._active_inference_keys()),
            "mlx_memory": get_mlx_memory_snapshot(),
            "prompt_cache": prompt_cache_stats,
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
