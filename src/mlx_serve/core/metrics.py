"""Prometheus metrics for mlx-serve."""

import time

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Request metrics
REQUESTS_TOTAL = Counter(
    "mlx_serve_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

REQUEST_DURATION = Histogram(
    "mlx_serve_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_SIZE = Histogram(
    "mlx_serve_request_size_bytes",
    "Request size in bytes",
    ["endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)

RESPONSE_SIZE = Histogram(
    "mlx_serve_response_size_bytes",
    "Response size in bytes",
    ["endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)

# Model metrics
MODEL_LOAD_DURATION = Histogram(
    "mlx_serve_model_load_seconds",
    "Time to load a model in seconds",
    ["model", "type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

MODEL_INFERENCE_DURATION = Histogram(
    "mlx_serve_model_inference_seconds",
    "Model inference time in seconds",
    ["model", "operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

MODELS_LOADED = Gauge(
    "mlx_serve_models_loaded",
    "Number of models currently loaded",
    ["type"],
)

MODEL_ESTIMATED_LOADED_BYTES = Gauge(
    "mlx_serve_model_estimated_loaded_bytes",
    "Estimated loaded model weight bytes",
    ["type"],
)

# Cache metrics
CACHE_HITS = Counter(
    "mlx_serve_cache_hits_total",
    "Total cache hits",
    ["type"],
)

CACHE_MISSES = Counter(
    "mlx_serve_cache_misses_total",
    "Total cache misses",
    ["type"],
)

CACHE_EVICTIONS = Counter(
    "mlx_serve_cache_evictions_total",
    "Total cache evictions",
    ["type"],
)

GENERATION_MEMORY_REJECTIONS = Counter(
    "mlx_serve_generation_memory_rejections_total",
    "Total generation requests rejected by the memory guard",
    ["type"],
)

GENERATION_MEMORY_REJECTED_KV_BYTES = Histogram(
    "mlx_serve_generation_memory_rejected_kv_bytes",
    "Estimated KV cache bytes for generation requests rejected by the memory guard",
    ["type"],
    buckets=(
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        32 * 1024 * 1024 * 1024,
    ),
)

MODEL_LOAD_MEMORY_REJECTIONS = Counter(
    "mlx_serve_model_load_memory_rejections_total",
    "Total model loads rejected by the memory guard",
    ["type"],
)

MODEL_LOAD_MEMORY_REJECTED_REQUIRED_BYTES = Histogram(
    "mlx_serve_model_load_memory_rejected_required_bytes",
    "Estimated required bytes for model loads rejected by the memory guard",
    ["type"],
    buckets=(
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        32 * 1024 * 1024 * 1024,
        64 * 1024 * 1024 * 1024,
    ),
)

# System metrics
MEMORY_USAGE = Gauge(
    "mlx_serve_memory_usage_bytes",
    "Memory usage in bytes",
)

MLX_ACTIVE_MEMORY = Gauge(
    "mlx_serve_mlx_active_memory_bytes",
    "MLX active memory in bytes",
)

MLX_CACHE_MEMORY = Gauge(
    "mlx_serve_mlx_cache_memory_bytes",
    "MLX cache memory in bytes",
)

MLX_PEAK_MEMORY = Gauge(
    "mlx_serve_mlx_peak_memory_bytes",
    "MLX peak memory in bytes",
)

ACTIVE_REQUESTS = Gauge(
    "mlx_serve_active_requests",
    "Number of active requests",
)

# Batch metrics
BATCH_SIZE = Histogram(
    "mlx_serve_batch_size",
    "Batch size distribution",
    ["endpoint"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

BATCH_WAIT_TIME = Histogram(
    "mlx_serve_batch_wait_seconds",
    "Time spent waiting for batch collection",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)


class MetricsMiddleware:
    """FastAPI middleware for collecting request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        path = scope.get("path", "unknown")
        method = scope.get("method", "unknown")

        # Track active requests
        ACTIVE_REQUESTS.inc()

        # Capture response status
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time

            # Record metrics
            REQUESTS_TOTAL.labels(
                endpoint=path,
                method=method,
                status=str(status_code),
            ).inc()

            REQUEST_DURATION.labels(
                endpoint=path,
                method=method,
            ).observe(duration)

            ACTIVE_REQUESTS.dec()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    refresh_runtime_metrics()
    return generate_latest()


def record_model_load(model: str, model_type: str, duration: float) -> None:
    """Record model loading metrics."""
    MODEL_LOAD_DURATION.labels(model=model, type=model_type).observe(duration)


def record_inference(model: str, operation: str, duration: float) -> None:
    """Record inference metrics."""
    MODEL_INFERENCE_DURATION.labels(model=model, operation=operation).observe(duration)


def record_cache_hit(cache_type: str) -> None:
    """Record a cache hit."""
    CACHE_HITS.labels(type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """Record a cache miss."""
    CACHE_MISSES.labels(type=cache_type).inc()


def record_cache_eviction(cache_type: str) -> None:
    """Record a cache eviction."""
    CACHE_EVICTIONS.labels(type=cache_type).inc()


def record_generation_memory_rejection(model_type: str, kv_cache_bytes: int) -> None:
    """Record a generation memory guard rejection."""
    GENERATION_MEMORY_REJECTIONS.labels(type=model_type).inc()
    GENERATION_MEMORY_REJECTED_KV_BYTES.labels(type=model_type).observe(
        max(0, int(kv_cache_bytes))
    )


def record_model_load_memory_rejection(model_type: str, required_bytes: int) -> None:
    """Record a model-load memory guard rejection."""
    MODEL_LOAD_MEMORY_REJECTIONS.labels(type=model_type).inc()
    MODEL_LOAD_MEMORY_REJECTED_REQUIRED_BYTES.labels(type=model_type).observe(
        max(0, int(required_bytes))
    )


def update_models_loaded(embedding_count: int, reranker_count: int) -> None:
    """Update the number of loaded models."""
    MODELS_LOADED.labels(type="embedding").set(embedding_count)
    MODELS_LOADED.labels(type="reranker").set(reranker_count)


def update_loaded_model_metrics(cache_stats: dict) -> None:
    """Update loaded model count and estimated-size gauges from cache stats."""
    for key, value in cache_stats.items():
        if not key.endswith("_models") or not isinstance(value, dict):
            continue
        model_type = key.removesuffix("_models")
        if model_type == "image_gen":
            metric_type = "image_gen"
        else:
            metric_type = model_type.removesuffix("_models")
        details = value.get("model_details", [])
        MODELS_LOADED.labels(type=metric_type).set(value.get("count", 0))
        MODEL_ESTIMATED_LOADED_BYTES.labels(type=metric_type).set(
            sum(int(item.get("estimated_weight_bytes") or 0) for item in details)
        )


def record_batch_size(endpoint: str, size: int) -> None:
    """Record batch size."""
    BATCH_SIZE.labels(endpoint=endpoint).observe(size)


def record_batch_wait(wait_time: float) -> None:
    """Record batch wait time."""
    BATCH_WAIT_TIME.observe(wait_time)


def update_memory_usage(usage_bytes: int) -> None:
    """Update memory usage."""
    MEMORY_USAGE.set(usage_bytes)


def update_mlx_memory(snapshot: dict) -> None:
    """Update MLX memory gauges from a runtime snapshot."""
    if not snapshot.get("available"):
        return
    active = snapshot.get("active_bytes")
    cache = snapshot.get("cache_bytes")
    peak = snapshot.get("peak_bytes")
    if active is not None:
        MLX_ACTIVE_MEMORY.set(int(active))
    if cache is not None:
        MLX_CACHE_MEMORY.set(int(cache))
    if peak is not None:
        MLX_PEAK_MEMORY.set(int(peak))


def refresh_runtime_metrics() -> None:
    """Refresh optional runtime gauges before metrics are rendered."""
    try:
        from mlx_serve.core.mlx_memory import get_mlx_memory_snapshot

        update_mlx_memory(get_mlx_memory_snapshot())
    except Exception:
        pass

    try:
        from mlx_serve.core.model_manager import model_manager

        update_loaded_model_metrics(model_manager.get_cache_stats())
    except Exception:
        pass
