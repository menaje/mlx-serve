"""Tests for Prometheus metric helpers."""

from prometheus_client import generate_latest


def test_update_mlx_memory_metrics():
    """MLX memory gauges should be exported when updated."""
    from mlx_serve.core import metrics

    metrics.update_mlx_memory(
        {
            "available": True,
            "active_bytes": 10,
            "cache_bytes": 20,
            "peak_bytes": 30,
        }
    )

    output = generate_latest().decode()
    assert "mlx_serve_mlx_active_memory_bytes 10.0" in output
    assert "mlx_serve_mlx_cache_memory_bytes 20.0" in output
    assert "mlx_serve_mlx_peak_memory_bytes 30.0" in output


def test_update_loaded_model_metrics():
    """Loaded model gauges should include counts and estimated bytes by type."""
    from mlx_serve.core import metrics

    metrics.update_loaded_model_metrics(
        {
            "llm_models": {
                "count": 2,
                "model_details": [
                    {"estimated_weight_bytes": 10},
                    {"estimated_weight_bytes": 20},
                ],
            }
        }
    )

    output = generate_latest().decode()
    assert 'mlx_serve_models_loaded{type="llm"} 2.0' in output
    assert 'mlx_serve_model_estimated_loaded_bytes{type="llm"} 30.0' in output


def test_record_generation_memory_rejection():
    """Generation memory rejections should expose count and estimated KV bytes."""
    from mlx_serve.core import metrics

    metrics.record_generation_memory_rejection("vlm", 123)

    output = generate_latest().decode()
    assert 'mlx_serve_generation_memory_rejections_total{type="vlm"}' in output
    assert 'mlx_serve_generation_memory_rejected_kv_bytes_count{type="vlm"}' in output
    assert 'mlx_serve_generation_memory_rejected_kv_bytes_sum{type="vlm"}' in output


def test_record_model_load_memory_rejection():
    """Model load memory rejections should expose count and required bytes."""
    from mlx_serve.core import metrics

    metrics.record_model_load_memory_rejection("llm", 456)

    output = generate_latest().decode()
    assert 'mlx_serve_model_load_memory_rejections_total{type="llm"}' in output
    assert 'mlx_serve_model_load_memory_rejected_required_bytes_count{type="llm"}' in output
    assert 'mlx_serve_model_load_memory_rejected_required_bytes_sum{type="llm"}' in output
