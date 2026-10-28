# mlx-serve

MLX-based multimodal AI server with OpenAI-compatible API for Apple Silicon.

## Features

- **OpenAI-compatible Chat API** (`/v1/chat/completions`) - LLM and VLM support
- **OpenAI-compatible Embeddings API** (`/v1/embeddings`)
- **Jina-compatible Reranking API** (`/v1/rerank`)
- **Text decisions** (`/v1/decisions`) with d1-omni, and Ollama-compatible
  `/v1/systemone` with d3-lite
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
- Python 3.11+

## Installation

### From source (recommended)

```bash
# Using pipx
pipx install git+https://github.com/menaje/mlx-serve.git

# For VLM model conversion, inject torch dependencies
pipx inject mlx-serve torch torchvision
```

### Using pip

```bash
pip install git+https://github.com/menaje/mlx-serve.git

# With VLM conversion support
pip install "mlx-serve[vlm-convert] @ git+https://github.com/menaje/mlx-serve.git"
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

# Import an Ollama MLX decision model already installed locally
mlx-serve import-ollama d3-lite:0.8b-mlx-bf16

# Or download the original decision checkpoint from Hugging Face
mlx-serve pull vllm-sr/d3-lite --type decision

# Download and convert the original d1 model to MLX FP16, then validate it
mlx-serve pull LiquidAI/d1-omni-600M --type decision

# Alternatively verify and install Liquid AI's already-converted FP16 model
mlx-serve pull LiquidAI/d1-omni-600M-MLX-fp16 --type decision

# Pin the source to a branch, tag, or commit
mlx-serve pull LiquidAI/d1-omni-600M --type decision --revision <revision>

# Install a separate named copy and retain the original source files
mlx-serve pull LiquidAI/d1-omni-600M --type decision --name d1-with-source --keep-original
```

Importing from Ollama uses hard links when the model directories are on the
same filesystem and copies the files otherwise. The imported model remains
available if the Ollama model is removed. The default model name is `d3-lite`;
use `--name` to select another name.

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

#### Text decisions

The d1 profile uses the original Hugging Face model, resolves its revision to a
commit, and prepares an FP16 MLX artifact in a separate process with the
official `mlx-vlm` converter. It loads both the original and the converted
model and checks their official predictions before publishing the result. The
already-converted official FP16 repository skips conversion and is still
checked and loaded before publication. Unsupported decision architectures and
quantization requests fail explicitly. A validated installation is reused for
the same repository and revision. The model's directory contains
`mlx_serve_decision_manifest.json` with file hashes, the source revision,
converter version and options, validation results, license, precision and
input limit. The decision preparation path never deletes the shared Hugging
Face cache. Pass `--keep-original` (or `keep_original: true` to `/api/pull`) on
the first source installation to retain a separate original copy under
`~/.mlx-serve/decision-originals`; otherwise the temporary source is removed
after verification. To retain a source that was not originally kept, install
under another local name using `--name`.

The CLI, `POST /api/pull`, and automatic downloads share this preparation path.
The API request accepts `name`, `type: "decision"`, an optional `revision`,
and an optional local `model_name`;
its newline-delimited events distinguish `resolving`, `downloading`, `checking`,
`converting`, `validating`, `success`, and `error`. Decision preparation uses
`mlx-vlm==0.7.10`. The default conversion process uses the server's Python;
set `MLX_SERVE_DECISION_PREPARE_PYTHON` to a separate Python executable with a
compatible installation to isolate its dependencies. Time, process RSS and
temporary disk limits can be adjusted through
`MLX_SERVE_DECISION_PREPARE_TIMEOUT_SECONDS`,
`MLX_SERVE_DECISION_PREPARE_MAX_MEMORY_BYTES`, and
`MLX_SERVE_DECISION_PREPARE_MAX_DISK_BYTES`. RSS is a process measurement and
does not separately account for all system or GPU allocations.

```bash
curl http://localhost:8000/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d1-omni",
    "state": {"message": "The customer was charged twice and wants a refund."},
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Payments and refunds", "technical": "Software errors"}
      },
      "refund": {"type": "bool", "instructions": "Is a refund requested?"},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["routine", "soon", "urgent"]}
    }
  }'
```

`choice` returns the selected option ID and each option's probability. `bool`
(also accepted as `noul`) returns a Boolean and the probability of true.
`score` returns the probability distribution, its expected numeric level, and
the level definitions. The response includes the installed model name, source
revision, and input/output token usage; output tokens are zero for this
non-generative inference. Requests accept text or JSON state, 1–64 questions,
2–255 choices, and 2–10 score levels. Image and video inputs are not supported.
Inputs that would be truncated by the d1 tokenizer are rejected with HTTP 422.
The default per-question limit is 4,096 tokens, adjustable with
`MLX_SERVE_DECISION_MAX_INPUT_TOKENS` up to the model's own limit. Requests
that exceed `MLX_SERVE_DECISION_EXECUTION_TIMEOUT_SECONDS` return HTTP 504;
the model stays reserved until the running operation actually finishes.

The existing Ollama-compatible endpoint also remains available:

```bash
curl http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d3-lite",
    "state": "The customer was charged twice and asks for a refund.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Payments and refunds", "technical": "Software errors"}
      },
      "refund": {"type": "noul", "instructions": "Is a refund requested?"}
    }
  }'
```

The endpoint supports `choice`, `noul` (a probability from 0 to 1), and
`score` questions. Each question runs separately against the same state. It
currently accepts text/JSON state only; image and video inputs are not yet
supported. The default maximum input is 4,096 tokens per question and can be
changed with `MLX_SERVE_DECISION_MAX_INPUT_TOKENS`, up to the model's own limit.

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
mlx-serve pull <model> --type <llm|vlm|embedding|reranker|decision>
mlx-serve pull <model> --type llm --quantize 4  # With 4-bit quantization

# Import an installed Ollama MLX decision model
mlx-serve import-ollama d3-lite:0.8b-mlx-bf16 [--name d3-lite]

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

batch:
  max_batch_size: 32
  max_embedding_texts: 32
  rerank_batch_max_documents: 4
  rerank_batch_max_tokens: 2048
  max_wait_ms: 50
  queue_timeout_seconds: 180

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
| `MLX_SERVE_EMBEDDING_BATCH_MAX_TEXTS` | `32` | Cap total texts merged into one embedding model call |
| `MLX_SERVE_RERANK_BATCH_MAX_DOCUMENTS` | `4` | Cap documents per rerank micro-batch |
| `MLX_SERVE_RERANK_BATCH_MAX_TOKENS` | `2048` | Cap padded token budget per rerank micro-batch |
| `MLX_SERVE_INFERENCE_QUEUE_TIMEOUT_SECONDS` | `180` | Wait longer before rejecting queued model work |

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
