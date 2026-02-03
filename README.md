# mlx-serve

MLX-based multimodal AI server with OpenAI-compatible API for Apple Silicon.

## Features

- **OpenAI-compatible Chat API** (`/v1/chat/completions`) - LLM and VLM support
- **OpenAI-compatible Embeddings API** (`/v1/embeddings`)
- **Jina-compatible Reranking API** (`/v1/rerank`)
- **Token counting API** (`/v1/tokenize`)
- **Ollama-compatible Model Management** (`/api/pull`, `/api/tags`, etc.)
- **Native Metal acceleration** on Apple Silicon
- **Model quantization** (4-bit, 8-bit) during conversion
- **Streaming support** for chat completions
- **Vision Language Models** (VLM) with image input support
- **YAML configuration file** support (`~/.mlx-serve/config.yaml`)
- **Prometheus metrics** (`/metrics` endpoint)
- **Auto-download** models on first request
- **Model aliases** for common models
- **CLI** for server and model management
- **System service integration** - launchd (macOS) and systemd (Linux)
- **Model caching** with LRU eviction and TTL-based expiration

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Installation

### Using Homebrew (macOS)

```bash
brew tap menaje/mlx-serve
brew install mlx-serve
```

### Using pip

```bash
pip install mlx-serve
```

### Using pipx (recommended)

```bash
pipx install mlx-serve

# For VLM model conversion, inject torch dependencies
pipx inject mlx-serve torch torchvision
```

## Quick Start

### 1. Download models

```bash
# Download an LLM model
mlx-serve pull Qwen/Qwen3-0.6B --type llm

# Download with 4-bit quantization
mlx-serve pull Qwen/Qwen3-1.7B --type llm --quantize 4

# Download a VLM model (requires torch/torchvision)
mlx-serve pull Qwen/Qwen3-VL-2B-Thinking --type vlm --quantize 4

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

# Start with model preloading
mlx-serve start --preload Qwen3-0.6B-4bit
```

### 3. Use the API

#### Chat Completions (LLM)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

#### Chat Completions with Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
  }'
```

#### Vision Language Model (VLM)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-2B-Thinking-4bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 200
  }'
```

#### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-0.6B",
    "input": ["Hello world", "How are you?"]
  }'
```

#### Reranking

```bash
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
mlx-serve start [--host 0.0.0.0] [--port 8000] [--foreground]
mlx-serve start --preload Qwen3-0.6B-4bit --preload Qwen3-Embedding-0.6B
mlx-serve status [--port 8000]
mlx-serve stop [--port 8000] [--force]
mlx-serve stop --all
```

### Model Management

```bash
# Pull models from Hugging Face
mlx-serve pull <model> --type <llm|vlm|embedding|reranker>
mlx-serve pull <model> --type llm --quantize 4  # With 4-bit quantization

# List installed models
mlx-serve list

# Remove a model
mlx-serve remove <model>

# Quantize an existing model
mlx-serve quantize <model> --bits 4
```

### Configuration

```bash
mlx-serve config                # Show current configuration
mlx-serve config --example      # Print example config file
mlx-serve config --path         # Show config file location
```

### Service Management (macOS launchd / Linux systemd)

```bash
mlx-serve service install
mlx-serve service start
mlx-serve service stop
mlx-serve service status
mlx-serve service enable     # Enable auto-start at login
mlx-serve service disable
mlx-serve service uninstall
```

## API Reference

### POST /v1/chat/completions

Generate chat completions (OpenAI-compatible).

**Request:**
```json
{
  "model": "Qwen3-0.6B-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["\n\n"]
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen3-0.6B-4bit",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

### POST /v1/embeddings

Create embeddings for text input (OpenAI-compatible).

**Request:**
```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["text1", "text2"],
  "encoding_format": "float",
  "dimensions": 256
}
```

**Parameters:**
- `encoding_format`: `"float"` (default) or `"base64"` for reduced response size
- `dimensions`: Truncate embeddings (requires MRL-trained model)

### POST /v1/rerank

Rerank documents by relevance to a query.

**Request:**
```json
{
  "model": "Qwen3-Reranker-0.6B",
  "query": "search query",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 3,
  "return_documents": true
}
```

### POST /v1/tokenize

Count tokens for text input.

**Request:**
```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["text1", "text2"],
  "return_tokens": false
}
```

### GET /v1/models

List available models (OpenAI format).

### GET /api/tags

List available models (Ollama format).

## Configuration

### YAML Configuration File

Create `~/.mlx-serve/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  directory: ~/.mlx-serve/models
  preload:
    - Qwen3-0.6B-4bit
  auto_download: true

cache:
  max_embedding_models: 3
  max_reranker_models: 2
  ttl_seconds: 1800

logging:
  level: INFO
  format: json
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SERVE_HOST` | `0.0.0.0` | Server host |
| `MLX_SERVE_PORT` | `8000` | Server port |
| `MLX_SERVE_MODELS_DIR` | `~/.mlx-serve/models` | Model storage path |
| `MLX_SERVE_LOG_LEVEL` | `INFO` | Log level |
| `MLX_SERVE_PRELOAD_MODELS` | `` | Comma-separated model names to preload |
| `MLX_SERVE_AUTO_DOWNLOAD` | `false` | Auto-download missing models |

## Benchmark Results

Token generation speed on Apple Silicon (M-series):

| Model | Precision | Avg TPS |
|-------|-----------|---------|
| Qwen3-0.6B | 4-bit | 148.32 |
| Qwen3-VL-2B-Thinking | 4-bit | 108.28 |
| Qwen3-1.7B | 4-bit | 69.76 |
| Qwen3-0.6B | Full | 59.10 |
| Qwen3-VL-2B-Thinking | Full | 32.47 |
| Qwen3-1.7B | Full | 23.13 |

4-bit quantization provides 2.5-3.3x speedup with minimal quality loss.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with VLM conversion support
pip install -e ".[vlm-convert]"

# Run tests
pytest

# Lint
ruff check .
```

## License

MIT
