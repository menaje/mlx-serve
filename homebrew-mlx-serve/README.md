# Homebrew Tap for mlx-serve

This is the official Homebrew tap for [mlx-serve](https://github.com/menaje/mlx-serve), an MLX-based embedding and reranking server with OpenAI-compatible API.

## Installation

```bash
brew tap menaje/mlx-serve
brew install mlx-serve
```

## Usage

### Start the server

```bash
# Run as a background service
brew services start mlx-serve

# Or run in foreground
mlx-serve start --foreground
```

### Stop the server

```bash
brew services stop mlx-serve
```

### Download a model

```bash
mlx-serve pull Qwen/Qwen3-Embedding-0.6B
```

### Configuration

Generate an example config file:

```bash
mlx-serve config --example > ~/.mlx-serve/config.yaml
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+

## Documentation

For more information, visit the [mlx-serve repository](https://github.com/menaje/mlx-serve).
