"""Tests for generation memory cleanup behavior."""

import pytest


def _model_load_error():
    from mlx_serve.core.model_memory import ModelLoadMemoryError, ModelMemoryEstimate

    return ModelLoadMemoryError(
        ModelMemoryEstimate(
            model_name="test-model",
            model_type="llm",
            estimated_weight_bytes=100,
            required_bytes=120,
            system_available_bytes=0,
            reserved_headroom_bytes=0,
        )
    )


def _generation_memory_error():
    from mlx_serve.core.model_memory import GenerationMemoryError, GenerationMemoryEstimate

    return GenerationMemoryError(
        GenerationMemoryEstimate(
            model_name="test-model",
            model_type="llm",
            prompt_tokens=10,
            max_tokens=10,
            kv_cache_bytes=1000,
            system_available_bytes=0,
            reserved_headroom_bytes=0,
        )
    )


def test_chat_completion_clears_mlx_cache(client, monkeypatch):
    """Non-streaming chat completions should clear MLX cache after generation."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    calls: list[str | None] = []

    async def fake_generate_completion(**kwargs):
        return "hello", 1, 1

    monkeypatch.setattr(settings, "generation_clear_mlx_cache_after_request", True)
    monkeypatch.setattr(
        chat_router.model_manager,
        "get_llm_model",
        lambda _model: (object(), object()),
    )
    monkeypatch.setattr(chat_router, "_generate_completion", fake_generate_completion)
    monkeypatch.setattr(
        chat_router,
        "clear_mlx_cache",
        lambda **kwargs: calls.append(kwargs.get("reason")) or True,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    assert "/v1/chat/completions" in calls


def test_chat_completion_returns_503_for_memory_load_failure(client, monkeypatch):
    """Model load memory failures should map to structured 503 responses."""
    from mlx_serve.routers import chat as chat_router

    monkeypatch.setattr(
        chat_router.model_manager,
        "get_llm_model",
        lambda _model: (_ for _ in ()).throw(_model_load_error()),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "server_overloaded"


def test_chat_completion_returns_503_for_generation_memory_failure(client, monkeypatch):
    """Request-scoped KV memory failures should map to structured 503 responses."""
    from mlx_serve.routers import chat as chat_router

    monkeypatch.setattr(
        chat_router.model_manager,
        "get_llm_model",
        lambda _model: (object(), object()),
    )
    monkeypatch.setattr(
        chat_router.model_manager,
        "check_generation_memory_available",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_generation_memory_error()),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "server_overloaded"
    assert "KV cache" in response.json()["detail"]["error"]["message"]


def test_vlm_generation_memory_includes_image_token_budget(monkeypatch):
    """VLM request memory checks should include configured image token reservations."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    calls = []

    class FakeTokenizer:
        def encode(self, _prompt):
            return [1, 2, 3]

    def fake_check_generation_memory_available(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(settings, "memory_vlm_image_tokens_per_image", 256)
    monkeypatch.setattr(
        chat_router.model_manager,
        "check_generation_memory_available",
        fake_check_generation_memory_available,
    )

    chat_router._check_generation_memory_for_request(
        "vlm",
        "test-vlm",
        FakeTokenizer(),
        "hello",
        max_tokens=5,
        image_count=2,
    )

    assert calls == [(("vlm", "test-vlm", 3, 5), {"image_tokens": 512})]


def test_preload_generation_memory_check_uses_text_and_image_budget(monkeypatch):
    """Pre-load memory checks should include rough prompt and VLM image tokens."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    calls = []

    def fake_ensure_load_and_generation_memory_available(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(settings, "memory_vlm_image_tokens_per_image", 128)
    monkeypatch.setattr(
        chat_router.model_manager,
        "ensure_load_and_generation_memory_available",
        fake_ensure_load_and_generation_memory_available,
    )

    chat_router._check_load_and_generation_memory_for_request(
        "vlm",
        "test-vlm",
        prompt_tokens=6,
        max_tokens=5,
        image_count=2,
    )

    assert calls == [(("vlm", "test-vlm", 6, 5), {"image_tokens": 256})]


def test_generation_kwargs_reflect_kv_quantization_settings(monkeypatch):
    """Generation helpers should pass configured KV quantization settings."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    monkeypatch.setattr(settings, "generation_kv_bits", 4)
    monkeypatch.setattr(settings, "generation_kv_group_size", 32)
    monkeypatch.setattr(settings, "generation_quantized_kv_start", 128)

    assert chat_router._generation_kwargs() == {
        "kv_bits": 4,
        "kv_group_size": 32,
        "quantized_kv_start": 128,
    }


def test_generation_kwargs_empty_when_kv_quantization_disabled(monkeypatch):
    """KV quantization should be opt-in."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    monkeypatch.setattr(settings, "generation_kv_bits", None)

    assert chat_router._generation_kwargs() == {}


def test_stream_stop_split_buffers_partial_stop_sequence():
    """Streaming stop handling should not emit partial stop sequences across chunks."""
    from mlx_serve.routers.chat import _split_stream_text_for_stop

    emit, pending, stopped = _split_stream_text_for_stop("hello ST", ["STOP"])
    assert emit == "hello "
    assert pending == "ST"
    assert stopped is False

    emit, pending, stopped = _split_stream_text_for_stop(pending + "OP ignored", ["STOP"])
    assert emit == ""
    assert pending == ""
    assert stopped is True


@pytest.mark.asyncio
async def test_generate_completion_does_not_commit_prompt_cache_on_stream_error(
    monkeypatch,
):
    """Prompt cache entries should not be retained after failed generation."""
    import sys
    import types

    import mlx_lm.models.cache as cache_module

    from mlx_serve.config import settings
    from mlx_serve.core.prompt_cache import prompt_cache_store
    from mlx_serve.routers import chat as chat_router

    trims = []

    class Tokenizer:
        def encode(self, text):
            return [1, 2, 3] if text == "prompt" else [4]

    class Chunk:
        text = "partial"

    def fake_stream_generate(*args, **kwargs):
        yield Chunk()
        raise RuntimeError("generation failed")

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(
            generate=lambda *args, **kwargs: "fallback",
            stream_generate=fake_stream_generate,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **kwargs: object()),
    )
    monkeypatch.setattr(settings, "generation_prompt_cache_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_max_entries", 4)
    monkeypatch.setattr(settings, "generation_prompt_cache_min_tokens", 2)
    monkeypatch.setattr(cache_module, "make_prompt_cache", lambda _model: ["cache"])
    monkeypatch.setattr(cache_module, "can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr(
        cache_module,
        "trim_prompt_cache",
        lambda _cache, count: trims.append(count) or count,
    )
    monkeypatch.setattr(
        chat_router.model_manager,
        "estimate_prompt_cache_bytes",
        lambda *_args: 300,
    )
    monkeypatch.setattr(
        chat_router.model_manager,
        "capture_memory_calibration_baseline",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        chat_router.model_manager,
        "calibrate_model_estimate_from_baseline",
        lambda *args: None,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        await chat_router._generate_completion(
            model=object(),
            tokenizer=Tokenizer(),
            model_name="model-a",
            prompt="prompt",
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )

    assert prompt_cache_store.stats()["count"] == 0
    assert trims == []


@pytest.mark.asyncio
async def test_vlm_generation_passes_multiple_images(monkeypatch):
    """VLM generation should pass all request images to mlx-vlm."""
    import sys
    import types

    from mlx_serve.routers import chat as chat_router

    calls = []

    class Result:
        text = "ok"

    def fake_generate(*args, **kwargs):
        calls.append(kwargs["image"])
        return Result()

    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(generate=fake_generate))
    monkeypatch.setattr(
        chat_router.model_manager,
        "capture_memory_calibration_baseline",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        chat_router.model_manager,
        "calibrate_model_estimate_from_baseline",
        lambda *args: None,
    )

    response, _, _ = await chat_router._generate_vlm_completion(
        model=object(),
        processor=object(),
        model_name="vlm",
        prompt="describe",
        images=["image-1", "image-2"],
        max_tokens=1,
        temperature=0.0,
    )

    assert response == "ok"
    assert calls == [["image-1", "image-2"]]


@pytest.mark.asyncio
async def test_chat_stream_clears_mlx_cache(monkeypatch):
    """Streaming chat finalizer should clear MLX cache."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    calls: list[str | None] = []

    class FakeLease:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_stream_completion(**kwargs):
        yield "chunk"

    monkeypatch.setattr(settings, "generation_clear_mlx_cache_after_request", True)
    monkeypatch.setattr(chat_router, "_stream_completion", fake_stream_completion)
    monkeypatch.setattr(
        chat_router,
        "clear_mlx_cache",
        lambda **kwargs: calls.append(kwargs.get("reason")) or True,
    )

    chunks = [
        chunk
        async for chunk in chat_router._stream_completion_with_lease(FakeLease())
    ]

    assert chunks == ["chunk"]
    assert "/v1/chat/completions stream" in calls


@pytest.mark.asyncio
async def test_text_completion_stream_clears_mlx_cache(monkeypatch):
    """Streaming completions finalizer should clear MLX cache."""
    from mlx_serve.config import settings
    from mlx_serve.routers import chat as chat_router

    calls: list[str | None] = []

    class FakeLease:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_stream_text_completion(**kwargs):
        yield "chunk"

    monkeypatch.setattr(settings, "generation_clear_mlx_cache_after_request", True)
    monkeypatch.setattr(chat_router, "_stream_text_completion", fake_stream_text_completion)
    monkeypatch.setattr(
        chat_router,
        "clear_mlx_cache",
        lambda **kwargs: calls.append(kwargs.get("reason")) or True,
    )

    chunks = [
        chunk
        async for chunk in chat_router._stream_text_completion_with_lease(FakeLease())
    ]

    assert chunks == ["chunk"]
    assert "/v1/completions stream" in calls


def test_image_generation_clears_mlx_cache(client, monkeypatch):
    """Image generation should clear MLX cache in its finalizer."""
    from mlx_serve.config import settings
    from mlx_serve.routers import images as images_router

    calls: list[str | None] = []

    async def fake_generate_image(**kwargs):
        return b"image"

    monkeypatch.setattr(settings, "generation_clear_mlx_cache_after_request", True)
    monkeypatch.setattr(
        images_router.model_manager,
        "get_image_gen_model",
        lambda _model: object(),
    )
    monkeypatch.setattr(images_router, "_generate_image", fake_generate_image)
    monkeypatch.setattr(
        images_router,
        "clear_mlx_cache",
        lambda **kwargs: calls.append(kwargs.get("reason")) or True,
    )

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "flux-schnell",
            "prompt": "test",
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    assert "/v1/images/generations" in calls
