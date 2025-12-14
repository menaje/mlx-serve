# mlx-serve

MLX-based embedding and reranking server with OpenAI-compatible API for Apple Silicon.

## Features

- **OpenAI-compatible API** for embeddings (`/v1/embeddings`)
- **Jina-compatible API** for reranking (`/v1/rerank`)
- **Ollama-compatible API** for model management (`/api/pull`, `/api/tags`, etc.)
- **Native Metal acceleration** on Apple Silicon
- **CLI** for server and model management
- **System service integration** - launchd (macOS) and systemd (Linux)
- **Model caching** with LRU eviction and TTL-based expiration
- **Server lifecycle control** - start, stop, status commands
- **Model preloading** for faster startup

## Requirements

- macOS with Apple Silicon (M1/M2/M3) or Linux
- Python 3.10+

## Installation

```bash
pip install mlx-serve
```

## Quick Start

### 1. Download a model

```bash
# Download an embedding model
mlx-serve pull Qwen/Qwen3-Embedding-0.6B --type embedding

# Download a reranker model
mlx-serve pull Qwen/Qwen3-Reranker-0.6B --type reranker
```

### 2. Start the server

```bash
# Start in foreground
mlx-serve start --foreground

# Start in background
mlx-serve start
```

### 3. Use the API

```bash
# Create embeddings (OpenAI-compatible)
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-0.6B",
    "input": ["Hello world", "How are you?"]
  }'

# Rerank documents (Jina-compatible)
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "What is machine learning?",
    "documents": ["ML is a subset of AI", "The weather is nice", "Deep learning uses neural networks"],
    "top_n": 2
  }'
```

## CLI Commands

### Server Management

```bash
# Start the server
mlx-serve start [--host 0.0.0.0] [--port 8000] [--foreground]

# Start with model preloading
mlx-serve start --preload Qwen3-Embedding-0.6B --preload Qwen3-Reranker-0.6B

# Check server status
mlx-serve status [--port 8000]

# Stop the server
mlx-serve stop [--port 8000] [--force]

# Stop all running instances
mlx-serve stop --all
```

### Model Management

```bash
mlx-serve pull <model> --type <embedding|reranker>
mlx-serve list
mlx-serve remove <model>
```

### Service Management (macOS launchd / Linux systemd)

```bash
mlx-serve service install    # Install as system service
mlx-serve service start      # Start service
mlx-serve service stop       # Stop service
mlx-serve service status     # Check status
mlx-serve service enable     # Enable auto-start at login
mlx-serve service disable    # Disable auto-start
mlx-serve service uninstall  # Remove service
```

## API Reference

### POST /v1/embeddings

Create embeddings for text input.

**Request:**
```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["text1", "text2"],
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [...], "index": 0},
    {"object": "embedding", "embedding": [...], "index": 1}
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 10, "total_tokens": 10}
}
```

### POST /v1/rerank

Rerank documents by relevance to a query.

**Request:**
```json
{
  "model": "Qwen3-Reranker-0.6B",
  "query": "search query",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 3,
  "return_documents": true,
  "return_text": false,
  "decision_threshold": 0.5
}
```

**Parameters:**
- `return_text` (bool, optional): Return "yes"/"no" text output based on relevance threshold. Default: `false`
- `decision_threshold` (float, optional): Threshold for yes/no decision (0.0-1.0). Default: `0.5`

**Response:**
```json
{
  "results": [
    {"index": 2, "relevance_score": 0.98, "document": {"text": "doc3"}, "text_output": "yes"},
    {"index": 0, "relevance_score": 0.76, "document": {"text": "doc1"}, "text_output": "yes"}
  ],
  "usage": {"prompt_tokens": 45, "total_tokens": 45}
}
```

### GET /v1/models

List available models (OpenAI format).

### GET /api/tags

List available models (Ollama format).

### POST /api/pull

Download and convert a model from Hugging Face.

### DELETE /api/delete

Delete an installed model.

### POST /api/show

Get detailed information about a model.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SERVE_HOST` | `0.0.0.0` | Server host |
| `MLX_SERVE_PORT` | `8000` | Server port |
| `MLX_SERVE_MODELS_DIR` | `~/.mlx-serve/models` | Model storage path |
| `MLX_SERVE_LOG_LEVEL` | `INFO` | Log level |
| `MLX_SERVE_CACHE_MAX_EMBEDDING_MODELS` | `3` | Max embedding models in cache (LRU) |
| `MLX_SERVE_CACHE_MAX_RERANKER_MODELS` | `2` | Max reranker models in cache (LRU) |
| `MLX_SERVE_CACHE_TTL_SECONDS` | `1800` | Model cache TTL in seconds (30 min) |
| `MLX_SERVE_PRELOAD_MODELS` | `` | Comma-separated model names to preload |

### Model Cache

mlx-serve uses an LRU (Least Recently Used) cache with TTL (Time-To-Live) for loaded models:

- **LRU eviction**: When the cache is full, the least recently used model is unloaded
- **TTL expiration**: Models unused for longer than the TTL are automatically unloaded
- **TTL refresh**: Accessing a model refreshes its TTL timer

This ensures efficient memory usage while maintaining fast response times for frequently used models.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## License

MIT
