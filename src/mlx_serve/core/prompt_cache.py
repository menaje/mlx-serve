"""Small in-process prompt KV cache store for MLX text generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_serve.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PromptCacheEntry:
    """Reusable prompt cache prefix."""

    model_name: str
    tokens: tuple[int, ...]
    cache: Any
    last_used: float
    estimated_bytes: int = 0
    hits: int = 0


@dataclass
class _PromptCacheTrieNode:
    """Prefix trie node for prompt cache entries."""

    children: dict[int, "_PromptCacheTrieNode"]
    entry: PromptCacheEntry | None = None


class PromptCacheLease:
    """Reserved mutable prompt cache for one generation request."""

    def __init__(
        self,
        store: "PromptCacheStore",
        model_name: str,
        full_tokens: tuple[int, ...],
        prompt_tokens: list[int],
        prompt_cache: Any,
        estimated_bytes: int,
        hit: bool,
    ) -> None:
        self._store = store
        self.model_name = model_name
        self.full_tokens = full_tokens
        self.prompt_tokens = prompt_tokens
        self.prompt_cache = prompt_cache
        self.estimated_bytes = max(0, int(estimated_bytes))
        self.hit = hit
        self._committed = False

    def commit(self, generated_tokens: int) -> None:
        """Trim generated tokens and return the prompt prefix to the store."""
        if self._committed:
            return
        self._committed = True
        self._store.store_after_generation(self, generated_tokens)


class PromptCacheStore:
    """Thread-safe exact-prefix prompt cache with LRU eviction."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, tuple[int, ...]], PromptCacheEntry] = {}
        self._tries: dict[str, _PromptCacheTrieNode] = {}
        self._lock = threading.RLock()

    def reserve(
        self,
        model_name: str,
        model: Any,
        token_ids: list[int],
        estimated_bytes: int = 0,
    ) -> PromptCacheLease | None:
        """Reserve a cache entry for the given prompt tokens."""
        if not self._enabled_for(token_ids):
            return None

        full_tokens = tuple(int(token) for token in token_ids)
        estimated_bytes = max(0, int(estimated_bytes))
        with self._lock:
            entry = self._find_best_prefix_entry(model_name, full_tokens)
            if entry is not None:
                self._remove_entry_locked(entry)
                entry.last_used = time.time()
                entry.hits += 1
                return PromptCacheLease(
                    self,
                    model_name,
                    full_tokens,
                    list(full_tokens[len(entry.tokens) :]),
                    entry.cache,
                    estimated_bytes or entry.estimated_bytes,
                    hit=True,
                )

            checkpoint = self._load_best_checkpoint_locked(
                model_name,
                full_tokens,
                estimated_bytes,
            )
            if checkpoint is not None:
                return checkpoint

        try:
            from mlx_lm.models.cache import make_prompt_cache

            prompt_cache = make_prompt_cache(model)
        except Exception:
            logger.debug("Failed to create MLX prompt cache", exc_info=True)
            return None

        return PromptCacheLease(
            self,
            model_name,
            full_tokens,
            list(full_tokens),
            prompt_cache,
            estimated_bytes,
            hit=False,
        )

    def store_after_generation(
        self,
        lease: PromptCacheLease,
        generated_tokens: int,
    ) -> None:
        """Store a reusable prefix after generation completes."""
        if not self._enabled_for(list(lease.full_tokens)) or len(lease.full_tokens) < 2:
            return

        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

            if not can_trim_prompt_cache(lease.prompt_cache):
                return
            # Leave the cache positioned at prompt[:-1]. The next request can pass
            # at least the final prompt token and let mlx-lm continue from there.
            trim_prompt_cache(lease.prompt_cache, max(0, int(generated_tokens)) + 1)
        except Exception:
            logger.debug("Failed to trim MLX prompt cache for reuse", exc_info=True)
            return

        cached_tokens = len(lease.full_tokens) - 1
        cached_bytes = lease.estimated_bytes
        if lease.full_tokens:
            cached_bytes = int(cached_bytes * (cached_tokens / len(lease.full_tokens)))

        entry = PromptCacheEntry(
            model_name=lease.model_name,
            tokens=lease.full_tokens[:-1],
            cache=lease.prompt_cache,
            last_used=time.time(),
            estimated_bytes=max(0, cached_bytes),
            hits=1 if lease.hit else 0,
        )
        if not entry.tokens:
            return

        with self._lock:
            self._insert_entry_locked(entry)
            self._prune_locked()
            self._save_checkpoint_locked(entry)
            self._prune_checkpoints_locked()

    def clear(self, model_name: str | None = None) -> int:
        """Clear cached prompt prefixes and return the number removed."""
        with self._lock:
            before = len(self._entries)
            if model_name is None:
                self._entries.clear()
                self._tries.clear()
            else:
                for key, entry in list(self._entries.items()):
                    if entry.model_name == model_name:
                        self._entries.pop(key, None)
                self._tries.pop(model_name, None)
            self._clear_checkpoints_locked(model_name)
            return before - len(self._entries)

    def stats(self) -> dict[str, object]:
        """Return cache diagnostics."""
        with self._lock:
            entries = sorted(
                self._entries.values(),
                key=lambda entry: entry.last_used,
                reverse=True,
            )
            estimated_bytes_total = sum(entry.estimated_bytes for entry in entries)
            checkpoint_dir = self._checkpoint_dir()
            checkpoint_count = len(list(checkpoint_dir.glob("*.json"))) if checkpoint_dir else 0
            return {
                "enabled": settings.generation_prompt_cache_enabled,
                "max_entries": settings.generation_prompt_cache_max_entries,
                "min_tokens": settings.generation_prompt_cache_min_tokens,
                "count": len(entries),
                "estimated_bytes": estimated_bytes_total,
                "prefix_index": "trie",
                "checkpoint_enabled": settings.generation_prompt_cache_checkpoint_enabled,
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "checkpoint_count": checkpoint_count,
                "entries": [
                    {
                        "model": entry.model_name,
                        "tokens": len(entry.tokens),
                        "last_used": entry.last_used,
                        "estimated_bytes": entry.estimated_bytes,
                        "hits": entry.hits,
                    }
                    for entry in entries
                ],
            }

    def _enabled_for(self, token_ids: list[int]) -> bool:
        return (
            settings.generation_prompt_cache_enabled
            and settings.generation_prompt_cache_max_entries > 0
            and len(token_ids) >= settings.generation_prompt_cache_min_tokens
        )

    def _entry_key(
        self,
        model_name: str,
        tokens: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...]]:
        return model_name, tokens

    def _insert_entry_locked(self, entry: PromptCacheEntry) -> None:
        existing = self._entries.get(self._entry_key(entry.model_name, entry.tokens))
        if existing is not None:
            self._remove_from_trie_locked(existing)
        self._entries[self._entry_key(entry.model_name, entry.tokens)] = entry
        node = self._tries.setdefault(entry.model_name, _PromptCacheTrieNode(children={}))
        for token in entry.tokens:
            node = node.children.setdefault(int(token), _PromptCacheTrieNode(children={}))
        node.entry = entry

    def _remove_entry_locked(self, entry: PromptCacheEntry) -> None:
        self._entries.pop(self._entry_key(entry.model_name, entry.tokens), None)
        self._remove_from_trie_locked(entry)

    def _remove_from_trie_locked(self, entry: PromptCacheEntry) -> None:
        root = self._tries.get(entry.model_name)
        if root is None:
            return

        path: list[tuple[_PromptCacheTrieNode, int]] = []
        node = root
        for token in entry.tokens:
            next_node = node.children.get(int(token))
            if next_node is None:
                return
            path.append((node, int(token)))
            node = next_node
        node.entry = None

        for parent, token in reversed(path):
            child = parent.children[token]
            if child.entry is not None or child.children:
                break
            parent.children.pop(token, None)

        if not root.children and root.entry is None:
            self._tries.pop(entry.model_name, None)

    def _find_best_prefix_entry(
        self,
        model_name: str,
        full_tokens: tuple[int, ...],
    ) -> PromptCacheEntry | None:
        node = self._tries.get(model_name)
        if node is None:
            return None

        best_entry: PromptCacheEntry | None = None
        for token in full_tokens[:-1]:
            node = node.children.get(int(token))
            if node is None:
                break
            if node.entry is not None:
                best_entry = node.entry
        return best_entry

    def _prune_locked(self) -> None:
        max_entries = settings.generation_prompt_cache_max_entries
        if len(self._entries) <= max_entries:
            return
        entries = sorted(
            self._entries.values(),
            key=lambda entry: entry.last_used,
            reverse=True,
        )
        for entry in entries[max_entries:]:
            self._remove_entry_locked(entry)

    def _checkpoint_dir(self) -> Path | None:
        if not settings.generation_prompt_cache_checkpoint_enabled:
            return None
        return settings.models_dir.parent / "prompt-cache"

    def _checkpoint_stem(self, model_name: str, tokens: tuple[int, ...]) -> str:
        payload = json.dumps(
            {"model": model_name, "tokens": list(tokens)},
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _checkpoint_paths(
        self,
        model_name: str,
        tokens: tuple[int, ...],
    ) -> tuple[Path, Path] | None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None:
            return None
        stem = self._checkpoint_stem(model_name, tokens)
        return checkpoint_dir / f"{stem}.safetensors", checkpoint_dir / f"{stem}.json"

    def _save_checkpoint_locked(self, entry: PromptCacheEntry) -> None:
        paths = self._checkpoint_paths(entry.model_name, entry.tokens)
        if paths is None:
            return
        cache_path, metadata_path = paths
        try:
            from mlx_lm.models.cache import save_prompt_cache

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache_path = cache_path.with_suffix(".safetensors.tmp")
            tmp_metadata_path = metadata_path.with_suffix(".json.tmp")
            metadata = {
                "model_name": entry.model_name,
                "tokens": list(entry.tokens),
                "token_count": len(entry.tokens),
                "estimated_bytes": entry.estimated_bytes,
                "last_used": entry.last_used,
                "hits": entry.hits,
            }
            save_prompt_cache(
                str(tmp_cache_path),
                entry.cache,
                metadata={key: str(value) for key, value in metadata.items()},
            )
            tmp_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            os.replace(tmp_cache_path, cache_path)
            os.replace(tmp_metadata_path, metadata_path)
        except Exception:
            logger.debug("Failed to save prompt cache checkpoint", exc_info=True)

    def _load_best_checkpoint_locked(
        self,
        model_name: str,
        full_tokens: tuple[int, ...],
        estimated_bytes: int,
    ) -> PromptCacheLease | None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return None

        best_metadata: dict[str, Any] | None = None
        best_tokens: tuple[int, ...] = ()
        for metadata_path in checkpoint_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("model_name") != model_name:
                continue
            raw_tokens = metadata.get("tokens")
            if not isinstance(raw_tokens, list):
                continue
            try:
                tokens = tuple(int(token) for token in raw_tokens)
            except (TypeError, ValueError):
                continue
            if len(tokens) >= len(full_tokens) or len(tokens) <= len(best_tokens):
                continue
            if full_tokens[: len(tokens)] != tokens:
                continue
            best_metadata = metadata
            best_tokens = tokens

        if best_metadata is None:
            return None

        paths = self._checkpoint_paths(model_name, best_tokens)
        if paths is None:
            return None
        cache_path, metadata_path = paths
        if not cache_path.exists():
            return None

        try:
            from mlx_lm.models.cache import load_prompt_cache

            prompt_cache = load_prompt_cache(str(cache_path))
        except Exception:
            logger.debug("Failed to load prompt cache checkpoint", exc_info=True)
            try:
                metadata_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        last_used = time.time()
        checkpoint_bytes = int(best_metadata.get("estimated_bytes") or 0)
        hits = int(best_metadata.get("hits") or 0) + 1
        best_metadata["last_used"] = last_used
        best_metadata["hits"] = hits
        try:
            metadata_path.write_text(json.dumps(best_metadata), encoding="utf-8")
        except OSError:
            pass

        return PromptCacheLease(
            self,
            model_name,
            full_tokens,
            list(full_tokens[len(best_tokens) :]),
            prompt_cache,
            estimated_bytes or checkpoint_bytes,
            hit=True,
        )

    def _clear_checkpoints_locked(self, model_name: str | None = None) -> None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return
        for metadata_path in checkpoint_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if model_name is not None and metadata.get("model_name") != model_name:
                continue
            cache_path = metadata_path.with_suffix(".safetensors")
            try:
                metadata_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed to remove prompt cache checkpoint", exc_info=True)

    def _prune_checkpoints_locked(self) -> None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return
        max_entries = settings.generation_prompt_cache_max_entries
        metadata_paths = sorted(
            checkpoint_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        for metadata_path in metadata_paths[max_entries:]:
            cache_path = metadata_path.with_suffix(".safetensors")
            try:
                metadata_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed to prune prompt cache checkpoint", exc_info=True)


prompt_cache_store = PromptCacheStore()
