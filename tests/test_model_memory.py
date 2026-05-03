"""Tests for model memory load admission helpers."""

import time

import pytest

from mlx_serve.core.system_guard import MemorySnapshot


def _snapshot(total: int, available: int) -> MemorySnapshot:
    return MemorySnapshot(
        process_rss_bytes=0,
        system_total_bytes=total,
        system_available_bytes=available,
        sampled_at=time.time(),
    )


def test_estimate_model_size_prefers_weight_files(tmp_path):
    """Weight files should drive the estimate when present."""
    from mlx_serve.core.model_memory import estimate_model_size_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    nested_dir = model_dir / "nested"
    nested_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"1" * 100)
    (nested_dir / "more.npz").write_bytes(b"3" * 50)
    (model_dir / "config.json").write_bytes(b"2" * 30)

    assert estimate_model_size_bytes(model_dir) == 150


def test_estimate_model_size_empty_directory(tmp_path):
    """Empty model directories should produce a zero estimate."""
    from mlx_serve.core.model_memory import estimate_model_size_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert estimate_model_size_bytes(model_dir) == 0


def test_estimate_model_size_falls_back_to_total_size(tmp_path):
    """Non-weight directories should still have a fallback estimate."""
    from mlx_serve.core.model_memory import estimate_model_size_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes(b"2" * 30)

    assert estimate_model_size_bytes(model_dir) == 30


def test_check_model_load_memory_rejects_when_estimate_exceeds_available(
    tmp_path,
    monkeypatch,
):
    """Load admission should reject models that exceed available memory."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import ModelLoadMemoryError, check_model_load_memory

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"1" * 100)

    monkeypatch.setattr(settings, "memory_load_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.10)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)
    monkeypatch.setattr(settings, "memory_model_size_multiplier", 1.20)

    with pytest.raises(ModelLoadMemoryError, match="insufficient memory"):
        check_model_load_memory(
            "tiny",
            "llm",
            model_dir,
            snapshot=_snapshot(total=1000, available=200),
        )


def test_check_model_load_memory_allows_when_estimate_fits(tmp_path, monkeypatch):
    """Load admission should pass when required bytes fit after headroom."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import check_model_load_memory

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"1" * 100)

    monkeypatch.setattr(settings, "memory_load_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.10)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)
    monkeypatch.setattr(settings, "memory_model_size_multiplier", 1.20)

    estimate = check_model_load_memory(
        "tiny",
        "llm",
        model_dir,
        snapshot=_snapshot(total=1000, available=300),
    )

    assert estimate.required_bytes == 120
    assert estimate.available_for_load_bytes == 200


def test_check_model_load_memory_includes_extra_required_bytes(tmp_path, monkeypatch):
    """Load admission should include expected request KV in required bytes."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import ModelLoadMemoryError, check_model_load_memory

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"1" * 100)

    monkeypatch.setattr(settings, "memory_load_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.0)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)
    monkeypatch.setattr(settings, "memory_model_size_multiplier", 1.0)

    with pytest.raises(ModelLoadMemoryError):
        check_model_load_memory(
            "tiny",
            "llm",
            model_dir,
            extra_required_bytes=50,
            snapshot=_snapshot(total=1000, available=120),
        )


def test_check_model_load_memory_allows_when_guard_disabled(tmp_path, monkeypatch):
    """Disabling the load guard should return the estimate without raising."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import check_model_load_memory

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"1" * 100)

    monkeypatch.setattr(settings, "memory_load_guard_enabled", False)

    estimate = check_model_load_memory(
        "tiny",
        "llm",
        model_dir,
        snapshot=_snapshot(total=1000, available=0),
    )

    assert estimate.estimated_weight_bytes == 100


def test_check_model_load_memory_rejects_explicit_remote_estimate(monkeypatch):
    """Remote loader estimates should use the same load admission path."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import ModelLoadMemoryError, check_model_load_memory

    monkeypatch.setattr(settings, "memory_load_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.0)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)
    monkeypatch.setattr(settings, "memory_model_size_multiplier", 1.0)

    with pytest.raises(ModelLoadMemoryError):
        check_model_load_memory(
            "remote",
            "image_gen",
            estimated_weight_bytes=1000,
            snapshot=_snapshot(total=2000, available=500),
        )


def test_remote_model_estimate_bytes_matches_repo_and_basename(monkeypatch):
    """Configured remote model estimates should match repo IDs and basename aliases."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import remote_model_estimate_bytes

    monkeypatch.setattr(
        settings,
        "memory_remote_model_estimates_bytes",
        {
            "org/model-a": 123,
            "model-b": 456,
        },
    )

    assert remote_model_estimate_bytes("org/model-a") == 123
    assert remote_model_estimate_bytes("other/model-b") == 456
    assert remote_model_estimate_bytes("missing") == 0


def test_estimate_generation_kv_cache_bytes_from_config(tmp_path, monkeypatch):
    """Generation KV estimates should use local model config dimensions."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_generation_kv_cache_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (
            '{"num_hidden_layers": 2, "num_attention_heads": 8, '
            '"num_key_value_heads": 4, "head_dim": 8}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 999)

    assert estimate_generation_kv_cache_bytes(model_dir, total_tokens=10) == 2560


def test_estimate_generation_kv_cache_bytes_uses_nested_text_config(
    tmp_path,
    monkeypatch,
):
    """VLM/converted configs should use nested language-model dimensions."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_generation_kv_cache_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (
            '{"model_type": "vlm", "text_config": {'
            '"num_hidden_layers": 2, "num_attention_heads": 8, '
            '"num_key_value_heads": 4, "head_dim": 8}}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 999)

    assert estimate_generation_kv_cache_bytes(model_dir, total_tokens=10) == 2560


def test_estimate_vlm_image_tokens_from_processor_config(tmp_path, monkeypatch):
    """VLM image token estimates should use processor patch and image size metadata."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_vlm_image_tokens

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text(
        '{"patch_size": 14, "size": {"height": 336, "width": 336}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_vlm_image_tokens_per_image", 1)

    assert estimate_vlm_image_tokens(model_dir, image_count=2) == 1152


def test_estimate_vlm_image_tokens_prefers_largest_edge(tmp_path, monkeypatch):
    """VLM image token estimates should be conservative for shortest/longest metadata."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_vlm_image_tokens

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text(
        '{"patch_size": 16, "size": {"shortest_edge": 224, "longest_edge": 448}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_vlm_image_tokens_per_image", 1)

    assert estimate_vlm_image_tokens(model_dir, image_count=1) == 784


def test_estimate_vlm_image_tokens_uses_nested_processor_config(tmp_path, monkeypatch):
    """VLM image estimates should read nested processor image metadata."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_vlm_image_tokens

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "processor_config.json").write_text(
        '{"image_processor": {"num_image_tokens": 1024}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_vlm_image_tokens_per_image", 1)

    assert estimate_vlm_image_tokens(model_dir, image_count=3) == 3072


def test_estimate_generation_kv_cache_bytes_uses_mla_config(
    tmp_path,
    monkeypatch,
):
    """MLA-style compressed KV config should drive KV estimates when present."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_generation_kv_cache_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"num_hidden_layers": 2, "kv_lora_rank": 8, "qk_rope_head_dim": 4}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 999)

    assert estimate_generation_kv_cache_bytes(model_dir, total_tokens=10) == 480


def test_estimate_generation_kv_cache_bytes_reflects_quantized_kv(
    tmp_path,
    monkeypatch,
):
    """KV estimates should shrink when quantized KV settings are enabled."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import estimate_generation_kv_cache_bytes

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (
            '{"num_hidden_layers": 2, "num_attention_heads": 8, '
            '"num_key_value_heads": 4, "head_dim": 8}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "generation_kv_bits", 4)
    monkeypatch.setattr(settings, "generation_kv_group_size", 64)
    monkeypatch.setattr(settings, "generation_quantized_kv_start", 2)

    assert estimate_generation_kv_cache_bytes(model_dir, total_tokens=10) == 1056


def test_check_generation_memory_rejects_when_kv_estimate_exceeds_available(
    tmp_path,
    monkeypatch,
):
    """Generation admission should reject requests with oversized KV estimates."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import GenerationMemoryError, check_generation_memory

    monkeypatch.setattr(settings, "memory_generation_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 100)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.0)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)

    with pytest.raises(GenerationMemoryError, match="KV cache"):
        check_generation_memory(
            "tiny",
            "llm",
            prompt_tokens=6,
            max_tokens=5,
            model_dir=tmp_path,
            snapshot=_snapshot(total=2000, available=1000),
        )


def test_check_generation_memory_allows_unknown_estimate(tmp_path, monkeypatch):
    """Unknown generation estimates should not block requests."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import check_generation_memory

    monkeypatch.setattr(settings, "memory_generation_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 0)

    estimate = check_generation_memory(
        "tiny",
        "llm",
        prompt_tokens=100,
        max_tokens=100,
        model_dir=tmp_path,
        snapshot=_snapshot(total=2000, available=0),
    )

    assert estimate.kv_cache_bytes == 0


def test_check_generation_memory_includes_image_tokens(tmp_path, monkeypatch):
    """VLM image token reservations should increase the KV estimate."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_memory import check_generation_memory

    monkeypatch.setattr(settings, "memory_generation_guard_enabled", True)
    monkeypatch.setattr(settings, "memory_generation_kv_bytes_per_token", 100)
    monkeypatch.setattr(settings, "memory_load_headroom_fraction", 0.0)
    monkeypatch.setattr(settings, "memory_min_free_bytes", None)

    estimate = check_generation_memory(
        "tiny-vlm",
        "vlm",
        prompt_tokens=3,
        max_tokens=2,
        image_tokens=4,
        model_dir=tmp_path,
        snapshot=_snapshot(total=2000, available=1000),
    )

    assert estimate.total_tokens == 9
    assert estimate.kv_cache_bytes == 900
