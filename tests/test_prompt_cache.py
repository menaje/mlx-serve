"""Tests for prompt KV cache reuse helpers."""

from pathlib import Path


def test_prompt_cache_reuses_longest_prefix(monkeypatch):
    """Prompt cache store should reserve and reuse exact token prefixes."""
    from mlx_serve.config import settings
    from mlx_serve.core.prompt_cache import PromptCacheStore

    trims: list[int] = []

    monkeypatch.setattr(settings, "generation_prompt_cache_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_max_entries", 2)
    monkeypatch.setattr(settings, "generation_prompt_cache_min_tokens", 3)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _model: {"state": []})
    monkeypatch.setattr("mlx_lm.models.cache.can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr(
        "mlx_lm.models.cache.trim_prompt_cache",
        lambda _cache, count: trims.append(count) or count,
    )

    store = PromptCacheStore()
    first = store.reserve("model-a", object(), [1, 2, 3, 4], estimated_bytes=400)

    assert first is not None
    assert first.hit is False
    assert first.prompt_tokens == [1, 2, 3, 4]

    first.commit(generated_tokens=2)
    second = store.reserve("model-a", object(), [1, 2, 3, 4, 5, 6], estimated_bytes=600)

    assert second is not None
    assert second.hit is True
    assert second.prompt_tokens == [4, 5, 6]
    assert trims == [3]

    second.commit(generated_tokens=1)
    stats = store.stats()
    assert stats["estimated_bytes"] == 500
    assert stats["prefix_index"] == "trie"
    assert stats["entries"][0]["estimated_bytes"] == 500


def test_prompt_cache_respects_entry_limit(monkeypatch):
    """Prompt cache store should keep only the configured number of entries."""
    from mlx_serve.config import settings
    from mlx_serve.core.prompt_cache import PromptCacheStore

    monkeypatch.setattr(settings, "generation_prompt_cache_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_max_entries", 1)
    monkeypatch.setattr(settings, "generation_prompt_cache_min_tokens", 3)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _model: [])
    monkeypatch.setattr("mlx_lm.models.cache.can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr("mlx_lm.models.cache.trim_prompt_cache", lambda _cache, count: count)

    store = PromptCacheStore()
    first = store.reserve("model-a", object(), [1, 2, 3])
    second = store.reserve("model-a", object(), [4, 5, 6])

    assert first is not None
    assert second is not None
    first.commit(generated_tokens=1)
    second.commit(generated_tokens=1)

    stats = store.stats()
    assert stats["count"] == 1


def test_prompt_cache_checkpoint_reuses_prefix_across_stores(tmp_path, monkeypatch):
    """Checkpoint-backed prompt cache should reload prefixes in a new store."""
    from mlx_serve.config import settings
    from mlx_serve.core.prompt_cache import PromptCacheStore

    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "generation_prompt_cache_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_checkpoint_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_max_entries", 4)
    monkeypatch.setattr(settings, "generation_prompt_cache_min_tokens", 3)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _model: ["cache"])
    monkeypatch.setattr("mlx_lm.models.cache.can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr("mlx_lm.models.cache.trim_prompt_cache", lambda _cache, count: count)

    def fake_save_prompt_cache(file_name: str, cache, metadata=None):
        Path(file_name).write_bytes(b"cache")

    def fake_load_prompt_cache(file_name: str):
        assert Path(file_name).exists()
        return ["loaded-cache"]

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", fake_save_prompt_cache)
    monkeypatch.setattr("mlx_lm.models.cache.load_prompt_cache", fake_load_prompt_cache)

    first_store = PromptCacheStore()
    first = first_store.reserve("model-a", object(), [1, 2, 3, 4], estimated_bytes=400)
    assert first is not None
    first.commit(generated_tokens=1)
    assert first_store.stats()["checkpoint_count"] == 1

    second_store = PromptCacheStore()
    second = second_store.reserve("model-a", object(), [1, 2, 3, 4, 5], estimated_bytes=500)

    assert second is not None
    assert second.hit is True
    assert second.prompt_tokens == [4, 5]
    assert second.prompt_cache == ["loaded-cache"]

    assert second_store.clear("model-a") == 0
    assert second_store.stats()["checkpoint_count"] == 0
