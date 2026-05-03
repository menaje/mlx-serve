"""Tests for models API."""


def test_list_models_openai(client):
    """Test OpenAI-compatible models list endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_list_models_ollama(client):
    """Test Ollama-compatible tags endpoint."""
    response = client.get("/api/tags")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_show_model_not_found(client):
    """Test show endpoint for non-existent model."""
    response = client.post("/api/show", json={"name": "nonexistent-model"})
    assert response.status_code == 404

    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"]["code"] == "model_not_found"


def test_delete_model_not_found(client):
    """Test delete endpoint for non-existent model."""
    response = client.request("DELETE", "/api/delete", json={"name": "nonexistent-model"})
    assert response.status_code == 404


def test_unload_model_endpoint(client, monkeypatch):
    """Unload endpoint should remove a cached model without deleting it."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._llm_cache.set("test-loaded", object())

    response = client.post("/v1/models/test-loaded/unload?type=llm")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["unloaded"] == [{"name": "test-loaded", "type": "llm"}]
    assert model_manager._llm_cache.get("test-loaded") is None


def test_unload_model_endpoint_keeps_model_type_backcompat(client, monkeypatch):
    """Unload endpoint should still accept the previous model_type query parameter."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._llm_cache.set("test-loaded", object())

    response = client.post("/v1/models/test-loaded/unload?model_type=llm")

    assert response.status_code == 200
    assert response.json()["unloaded"] == [{"name": "test-loaded", "type": "llm"}]


def test_unload_model_endpoint_returns_409_for_active_model(client, monkeypatch):
    """Unload endpoint should reject models with active inference leases."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._llm_cache.set("active-model", object())
    monkeypatch.setattr(model_manager, "_active_inference_keys", lambda: {"llm:active-model"})

    response = client.post("/v1/models/active-model/unload?type=llm")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "model_in_use"
    assert model_manager._llm_cache.get("active-model") is not None


def test_unload_all_endpoint(client, monkeypatch):
    """Unload-all endpoint should remove cached idle models."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._embedding_cache.set("embed-loaded", object())
    model_manager._reranker_cache.set("rerank-loaded", object())

    response = client.post("/v1/models/unload", json={"all": True})

    assert response.status_code == 200
    data = response.json()
    assert {"name": "embed-loaded", "type": "embedding"} in data["unloaded"]
    assert {"name": "rerank-loaded", "type": "reranker"} in data["unloaded"]


def test_unload_all_endpoint_filters_by_type(client, monkeypatch):
    """Unload-all should honor type filters."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._embedding_cache.set("embed-loaded", object())
    model_manager._llm_cache.set("llm-loaded", object())

    response = client.post("/v1/models/unload", json={"all": True, "type": "llm"})

    assert response.status_code == 200
    data = response.json()
    assert data["unloaded"] == [{"name": "llm-loaded", "type": "llm"}]
    assert model_manager._embedding_cache.get("embed-loaded") is not None


def test_idle_eviction_skips_active_models(tmp_path, monkeypatch):
    """Memory-pressure eviction should not unload active models."""
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.model_memory import ModelLoadMemoryError, ModelMemoryEstimate

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._llm_cache.set("active", object())
    model_manager._embedding_cache.set("idle", object())
    model_manager._model_estimated_weight_bytes[("llm", "active")] = 200
    model_manager._model_estimated_weight_bytes[("embedding", "idle")] = 100
    monkeypatch.setattr(model_manager, "_active_inference_keys", lambda: {"llm:active"})

    calls = {"count": 0}

    def fake_check_model_load_memory(
        model_name,
        model_type,
        model_dir=None,
        estimated_weight_bytes=None,
        extra_required_bytes=0,
    ):
        calls["count"] += 1
        estimate = ModelMemoryEstimate(
            model_name=model_name,
            model_type=model_type,
            estimated_weight_bytes=50,
            required_bytes=60,
            system_available_bytes=0,
            reserved_headroom_bytes=0,
        )
        if calls["count"] == 1:
            raise ModelLoadMemoryError(estimate)
        return estimate

    monkeypatch.setattr(
        "mlx_serve.core.model_manager.check_model_load_memory",
        fake_check_model_load_memory,
    )

    model_manager._ensure_load_memory_available("llm", "target", tmp_path)

    assert model_manager._llm_cache.get("active") is not None
    assert model_manager._embedding_cache.get("idle") is None
    assert ("embedding", "idle") not in model_manager._model_estimated_weight_bytes


def test_load_memory_pressure_clears_prompt_cache_before_model_eviction(
    tmp_path,
    monkeypatch,
):
    """Prompt KV cache should be released before evicting loaded models."""
    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.model_memory import ModelLoadMemoryError, ModelMemoryEstimate
    from mlx_serve.core.prompt_cache import prompt_cache_store

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(settings, "generation_prompt_cache_enabled", True)
    monkeypatch.setattr(settings, "generation_prompt_cache_max_entries", 4)
    monkeypatch.setattr(settings, "generation_prompt_cache_min_tokens", 2)
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda _model: [])
    monkeypatch.setattr("mlx_lm.models.cache.can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr("mlx_lm.models.cache.trim_prompt_cache", lambda _cache, count: count)

    model_manager.unload_all()
    model_manager._embedding_cache.set("idle", object())
    prompt_lease = prompt_cache_store.reserve("target", object(), [1, 2, 3])
    assert prompt_lease is not None
    prompt_lease.commit(generated_tokens=1)
    assert prompt_cache_store.stats()["count"] == 1

    calls = {"count": 0}

    def fake_check_model_load_memory(
        model_name,
        model_type,
        model_dir=None,
        estimated_weight_bytes=None,
        extra_required_bytes=0,
    ):
        calls["count"] += 1
        estimate = ModelMemoryEstimate(
            model_name=model_name,
            model_type=model_type,
            estimated_weight_bytes=50,
            required_bytes=60,
            system_available_bytes=0,
            reserved_headroom_bytes=0,
        )
        if calls["count"] == 1:
            raise ModelLoadMemoryError(estimate)
        return estimate

    monkeypatch.setattr(
        "mlx_serve.core.model_manager.check_model_load_memory",
        fake_check_model_load_memory,
    )

    model_manager._ensure_load_memory_available("llm", "target", tmp_path)

    assert calls["count"] == 2
    assert prompt_cache_store.stats()["count"] == 0
    assert model_manager._embedding_cache.get("idle") is not None


def test_combined_load_generation_guard_passes_kv_bytes_to_load_guard(
    tmp_path,
    monkeypatch,
):
    """Pre-load admission should include estimated request KV bytes."""
    from mlx_serve.core.model_manager import model_manager

    model_dir = tmp_path / "target"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        (
            '{"num_hidden_layers": 2, "num_attention_heads": 8, '
            '"num_key_value_heads": 4, "head_dim": 8}'
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_ensure(model_type, model_name, model_dir=None, **kwargs):
        calls.append((model_type, model_name, model_dir, kwargs))

    monkeypatch.setattr(model_manager, "_get_model_dir", lambda _name: model_dir)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", fake_ensure)

    model_manager.ensure_load_and_generation_memory_available(
        "llm",
        "target",
        prompt_tokens=3,
        max_tokens=2,
        image_tokens=1,
    )

    assert calls == [
        (
            "llm",
            "target",
            model_dir,
            {"extra_required_bytes": 1536},
        )
    ]


def test_generation_memory_eviction_retries_and_skips_active_models(
    tmp_path,
    monkeypatch,
):
    """Generation memory pressure should evict idle models before returning 503."""
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.model_memory import GenerationMemoryError, GenerationMemoryEstimate

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr("mlx_serve.core.metrics.record_cache_eviction", lambda _type: None)
    model_manager.unload_all()
    model_manager._llm_cache.set("target", object())
    model_manager._embedding_cache.set("active", object())
    model_manager._embedding_cache.set("idle", object())
    model_manager._model_estimated_weight_bytes[("llm", "target")] = 50
    model_manager._model_estimated_weight_bytes[("embedding", "active")] = 200
    model_manager._model_estimated_weight_bytes[("embedding", "idle")] = 100
    monkeypatch.setattr(model_manager, "_active_inference_keys", lambda: {"embedding:active"})
    monkeypatch.setattr(model_manager, "_get_model_dir", lambda _name: tmp_path)

    calls = {"count": 0}

    def fake_check_generation_memory(
        model_name,
        model_type,
        prompt_tokens,
        max_tokens,
        image_tokens=0,
        model_dir=None,
    ):
        calls["count"] += 1
        estimate = GenerationMemoryEstimate(
            model_name=model_name,
            model_type=model_type,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            image_tokens=image_tokens,
            kv_cache_bytes=1000,
            system_available_bytes=0,
            reserved_headroom_bytes=0,
        )
        if calls["count"] == 1:
            raise GenerationMemoryError(estimate)
        return estimate

    monkeypatch.setattr(
        "mlx_serve.core.model_manager.check_generation_memory",
        fake_check_generation_memory,
    )

    model_manager.check_generation_memory_available(
        "llm",
        "target",
        prompt_tokens=3,
        max_tokens=2,
        image_tokens=4,
    )

    assert calls["count"] == 2
    assert model_manager._llm_cache.get("target") is not None
    assert model_manager._embedding_cache.get("active") is not None
    assert model_manager._embedding_cache.get("idle") is None


def test_model_cache_stats_endpoint(client):
    """Cache stats endpoint should expose memory diagnostics."""
    response = client.get("/v1/models/cache")

    assert response.status_code == 200
    data = response.json()
    assert "ttl_seconds" in data
    assert "active_inference" in data
    assert "mlx_memory" in data


def test_cache_stats_uses_stored_estimates(client, monkeypatch):
    """Cache stats should use stored estimates instead of rescanning model files."""
    from mlx_serve.core.model_manager import model_manager

    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    model_manager.unload_all()
    model_manager._llm_cache.set("estimated-model", object())
    model_manager._model_estimated_weight_bytes[("llm", "estimated-model")] = 123

    response = client.get("/v1/models/cache")

    assert response.status_code == 200
    details = response.json()["llm_models"]["model_details"]
    assert details == [
        {
            "name": "estimated-model",
            "type": "llm",
            "last_used": details[0]["last_used"],
            "estimated_weight_bytes": 123,
        }
    ]


def test_image_gen_remote_alias_uses_configured_memory_estimate(
    tmp_path,
    monkeypatch,
):
    """mflux alias loads should run load admission with configured remote estimates."""
    import sys
    import types

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager

    class FakeFlux:
        @staticmethod
        def from_alias(alias):
            return {"alias": alias}

    calls = []

    def fake_ensure(model_type, model_name, model_dir=None, estimated_weight_bytes=None):
        calls.append((model_type, model_name, model_dir, estimated_weight_bytes))

    monkeypatch.setitem(sys.modules, "mflux", types.SimpleNamespace(Flux1=FakeFlux))
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(
        settings,
        "memory_remote_model_estimates_bytes",
        {"black-forest-labs/FLUX.1-schnell": 321},
    )
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", fake_ensure)

    model_manager.unload_all()
    model = model_manager.get_image_gen_model("flux-schnell")

    assert model == {"alias": "flux1-schnell"}
    assert calls == [("image_gen", "FLUX.1-schnell", None, 321)]


def test_image_gen_remote_alias_calibrates_estimate_from_observed_rss(
    tmp_path,
    monkeypatch,
):
    """Remote model estimates should be raised from observed load-time deltas."""
    import sys
    import time
    import types

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.system_guard import MemorySnapshot

    class FakeFlux:
        @staticmethod
        def from_alias(alias):
            return {"alias": alias}

    snapshots = iter(
        [
            MemorySnapshot(
                process_rss_bytes=1_000,
                system_total_bytes=10_000,
                system_available_bytes=9_000,
                sampled_at=time.time(),
            ),
            MemorySnapshot(
                process_rss_bytes=1_900,
                system_total_bytes=10_000,
                system_available_bytes=8_000,
                sampled_at=time.time(),
            ),
        ]
    )

    monkeypatch.setitem(sys.modules, "mflux", types.SimpleNamespace(Flux1=FakeFlux))
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(
        settings,
        "memory_remote_model_estimates_bytes",
        {"black-forest-labs/FLUX.1-schnell": 100},
    )
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.get_mlx_memory_snapshot",
        lambda: {"available": False},
    )
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.collect_memory_snapshot",
        lambda: next(snapshots),
    )

    model_manager.unload_all()
    model_manager.get_image_gen_model("flux-schnell")

    assert model_manager._model_estimated_weight_bytes[("image_gen", "FLUX.1-schnell")] == 900


def test_llm_local_loader_calibrates_estimate_from_observed_rss(
    tmp_path,
    monkeypatch,
):
    """Local LLM loads should raise stored estimates from observed load deltas."""
    import sys
    import time
    import types

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.system_guard import MemorySnapshot

    model_dir = tmp_path / "local-llm"
    model_dir.mkdir()
    snapshots = iter(
        [
            MemorySnapshot(1_000, 10_000, 9_000, time.time()),
            MemorySnapshot(1_700, 10_000, 8_000, time.time()),
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(load=lambda _path: ("model", "tokenizer")),
    )
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.get_mlx_memory_snapshot",
        lambda: {"available": False},
    )
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.collect_memory_snapshot",
        lambda: next(snapshots),
    )

    model_manager.unload_all()
    model_manager.get_llm_model("local-llm")

    assert model_manager._model_estimated_weight_bytes[("llm", "local-llm")] == 700


def test_vlm_local_loader_calibrates_estimate_from_observed_rss(
    tmp_path,
    monkeypatch,
):
    """Local VLM loads should raise stored estimates from observed load deltas."""
    import sys
    import time
    import types

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.system_guard import MemorySnapshot

    model_dir = tmp_path / "local-vlm"
    model_dir.mkdir()
    snapshots = iter(
        [
            MemorySnapshot(2_000, 10_000, 8_000, time.time()),
            MemorySnapshot(3_100, 10_000, 7_000, time.time()),
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(load=lambda _path: ("model", "processor")),
    )
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.get_mlx_memory_snapshot",
        lambda: {"available": False},
    )
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.collect_memory_snapshot",
        lambda: next(snapshots),
    )

    model_manager.unload_all()
    model_manager.get_vlm_model("local-vlm")

    assert model_manager._model_estimated_weight_bytes[("vlm", "local-vlm")] == 1100


def test_other_local_loaders_calibrate_estimate_from_observed_rss(
    tmp_path,
    monkeypatch,
):
    """Embedding, reranker, TTS, and STT loads should also calibrate estimates."""
    import sys
    import time
    import types

    from mlx_serve.config import settings
    from mlx_serve.core.model_manager import model_manager
    from mlx_serve.core.system_guard import MemorySnapshot

    def install_module(module_name, load_result):
        module = types.ModuleType(module_name)
        module.load = lambda _path: load_result
        monkeypatch.setitem(sys.modules, module_name, module)
        return module

    def install_audio_module(module_name, load_result):
        audio_pkg = sys.modules.get("mlx_audio") or types.ModuleType("mlx_audio")
        audio_pkg.__path__ = []
        module = install_module(module_name, load_result)
        setattr(audio_pkg, module_name.rsplit(".", 1)[-1], module)
        monkeypatch.setitem(sys.modules, "mlx_audio", audio_pkg)

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("mlx_serve.core.model_manager.clear_mlx_cache", lambda **kwargs: True)
    monkeypatch.setattr(model_manager, "_ensure_load_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "mlx_serve.core.model_manager.get_mlx_memory_snapshot",
        lambda: {"available": False},
    )

    cases = [
        (
            "embedding",
            "local-embedding",
            model_manager.get_embedding_model,
            lambda: install_module("mlx_embeddings", ("embedding-model", "tokenizer")),
            ("embedding-model", "tokenizer"),
            600,
        ),
        (
            "reranker",
            "local-reranker",
            model_manager.get_reranker_model,
            lambda: install_module("mlx_lm", ("reranker-model", "tokenizer")),
            ("reranker-model", "tokenizer"),
            700,
        ),
        (
            "tts",
            "local-tts",
            model_manager.get_tts_model,
            lambda: install_audio_module("mlx_audio.tts", "tts-model"),
            "tts-model",
            800,
        ),
        (
            "stt",
            "local-stt",
            model_manager.get_stt_model,
            lambda: install_audio_module("mlx_audio.stt", "stt-model"),
            "stt-model",
            900,
        ),
    ]

    for model_type, model_name, loader, install_loader, expected_model, delta in cases:
        model_dir = tmp_path / model_name
        model_dir.mkdir()
        install_loader()
        snapshots = iter(
            [
                MemorySnapshot(1_000, 10_000, 9_000, time.time()),
                MemorySnapshot(1_000 + delta, 10_000, 8_000, time.time()),
            ]
        )

        def next_snapshot(snapshot_iter=snapshots):
            return next(snapshot_iter)

        monkeypatch.setattr(
            "mlx_serve.core.model_manager.collect_memory_snapshot",
            next_snapshot,
        )

        model_manager.unload_all()
        model_manager._model_estimated_weight_bytes.clear()
        model = loader(model_name)

        assert model == expected_model
        assert model_manager._model_estimated_weight_bytes[(model_type, model_name)] == delta


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "mlx_memory" in data
