# Retrieval Worker Update

## Summary

This change replaces single-process retrieval execution with a gateway plus dedicated retrieval workers.

- gateway process: receives external API traffic
- embedding worker: serves `/v1/embeddings` and embedding-side `/v1/tokenize`
- reranker worker: serves `/v1/rerank` and reranker-side `/v1/tokenize`

The goal is to prevent cross-model MLX/Metal crashes when embedding and rerank requests arrive concurrently.

## Runtime Changes

### Retrieval isolation

By default, retrieval worker isolation is enabled.

- `POST /v1/embeddings` is proxied to the embedding worker
- `POST /v1/rerank` is proxied to the reranker worker
- `POST /v1/tokenize` is routed to the matching retrieval worker

### Tokenize routing

`/v1/tokenize` now resolves the retrieval model type before execution.

- If alias or metadata identifies the model type, the request is sent directly to the matching worker
- If the type cannot be inferred, the gateway probes the embedding worker and reranker worker in sequence
- Worker-local tokenize handling also allows legacy retrieval models that are present locally even when aliases or metadata are missing

### Shutdown behavior

When the gateway exits, managed retrieval workers are terminated as part of application shutdown.

## Configuration Additions

```yaml
retrieval:
  worker_isolation_enabled: true
  worker_host: 127.0.0.1
  worker_ready_timeout_seconds: 30.0
  worker_shutdown_timeout_seconds: 5.0
```

Meaning:

- `retrieval.worker_isolation_enabled`: enable gateway plus retrieval worker topology
- `retrieval.worker_host`: bind host for internal workers
- `retrieval.worker_ready_timeout_seconds`: worker health wait timeout
- `retrieval.worker_shutdown_timeout_seconds`: worker shutdown wait timeout

To force the legacy single-process retrieval path:

```yaml
retrieval:
  worker_isolation_enabled: false
```

## API Impact

### Stable external contracts

There is no request or response schema change for:

- `POST /v1/embeddings`
- `POST /v1/rerank`
- `POST /v1/tokenize`

The API contract remains the same. The change is internal routing and process topology.

### Health response expansion

`GET /health` now includes topology information.

Added fields:

- `role`
- `retrieval_worker_kind`
- `retrieval_workers` on the gateway

Example shape:

```json
{
  "role": "gateway",
  "retrieval_worker_kind": null,
  "retrieval_workers": {
    "embedding": {
      "url": "http://127.0.0.1:52562",
      "pid": 73371,
      "alive": true
    },
    "reranker": {
      "url": "http://127.0.0.1:52568",
      "pid": 73375,
      "alive": true
    }
  }
}
```

These worker URLs are internal runtime details, not public endpoints.

## Validation

Verified by tests:

```bash
pytest project/tests/test_server_topology.py \
       project/tests/test_tokenize.py \
       project/tests/test_embeddings.py \
       project/tests/test_rerank.py \
       project/tests/test_inference_control.py \
       project/tests/test_config.py
```

Observed result:

- `45 passed`

Verified manually:

- embedding plus rerank concurrent requests returned `200/200`
- embedding tokenize plus rerank tokenize concurrent requests returned `200/200`
- gateway remained alive after both scenarios
- gateway shutdown terminated both retrieval workers

## Main Files

- `src/mlx_serve/server.py`
- `src/mlx_serve/routers/retrieval_proxy.py`
- `src/mlx_serve/routers/tokenize.py`
- `src/mlx_serve/core/retrieval_workers.py`
- `src/mlx_serve/core/runtime_topology.py`
- `src/mlx_serve/core/retrieval_model_routing.py`
- `src/mlx_serve/config.py`
- `src/mlx_serve/core/config_loader.py`
