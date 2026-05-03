"""Model memory estimation and load admission helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlx_serve.config import settings
from mlx_serve.core.system_guard import MemorySnapshot, collect_memory_snapshot

WEIGHT_FILE_SUFFIXES = {
    ".safetensors",
    ".npz",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
}


def _format_bytes(value: int | None) -> str:
    """Render bytes in a compact human-readable form."""
    if value is None:
        return "unknown"

    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.1f}{unit}"


@dataclass(frozen=True)
class ModelMemoryEstimate:
    """Best-effort model load memory estimate."""

    model_name: str
    model_type: str
    estimated_weight_bytes: int
    required_bytes: int
    system_available_bytes: int | None
    reserved_headroom_bytes: int

    @property
    def available_for_load_bytes(self) -> int | None:
        """Return available memory after configured headroom is reserved."""
        if self.system_available_bytes is None:
            return None
        return max(0, self.system_available_bytes - self.reserved_headroom_bytes)


class ModelLoadMemoryError(RuntimeError):
    """Raised when a model load is expected to exceed available memory."""

    def __init__(self, estimate: ModelMemoryEstimate):
        self.estimate = estimate
        super().__init__(
            "insufficient memory to load "
            f"{estimate.model_type} model '{estimate.model_name}': "
            f"requires approximately {_format_bytes(estimate.required_bytes)}, "
            f"available after headroom is "
            f"{_format_bytes(estimate.available_for_load_bytes)}"
        )


@dataclass(frozen=True)
class GenerationMemoryEstimate:
    """Best-effort KV cache memory estimate for one generation request."""

    model_name: str
    model_type: str
    prompt_tokens: int
    max_tokens: int
    kv_cache_bytes: int
    system_available_bytes: int | None
    reserved_headroom_bytes: int
    image_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return prompt plus maximum generated tokens."""
        return self.prompt_tokens + self.max_tokens + self.image_tokens

    @property
    def available_for_request_bytes(self) -> int | None:
        """Return available memory after configured headroom is reserved."""
        if self.system_available_bytes is None:
            return None
        return max(0, self.system_available_bytes - self.reserved_headroom_bytes)


class GenerationMemoryError(RuntimeError):
    """Raised when generation KV cache memory is expected to exceed available memory."""

    def __init__(self, estimate: GenerationMemoryEstimate):
        self.estimate = estimate
        super().__init__(
            "insufficient memory for "
            f"{estimate.model_type} generation on '{estimate.model_name}': "
            f"requires approximately {_format_bytes(estimate.kv_cache_bytes)} "
            f"for {estimate.total_tokens} tokens of KV cache, "
            f"available after headroom is "
            f"{_format_bytes(estimate.available_for_request_bytes)}"
        )


def estimate_model_size_bytes(model_dir: Path) -> int:
    """Estimate model weight bytes from a local model directory."""
    if not model_dir.exists():
        return 0

    weight_bytes = 0
    total_bytes = 0
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total_bytes += size
        if path.suffix.lower() in WEIGHT_FILE_SUFFIXES:
            weight_bytes += size

    return weight_bytes or total_bytes


def remote_model_estimate_bytes(*names: str | None) -> int:
    """Return a configured estimate for loader-managed remote models."""
    estimates = settings.memory_remote_model_estimates_bytes or {}
    for name in names:
        if not name:
            continue
        direct = estimates.get(name)
        if direct is not None:
            return int(direct)
        base = name.rsplit("/", maxsplit=1)[-1]
        base_match = estimates.get(base)
        if base_match is not None:
            return int(base_match)
    return 0


def _estimate_required_bytes(estimated_weight_bytes: int) -> int:
    multiplier = max(1.0, float(settings.memory_model_size_multiplier))
    return int(estimated_weight_bytes * multiplier)


def _reserved_headroom_bytes(snapshot: MemorySnapshot) -> int:
    reserves: list[int] = []
    if settings.memory_min_free_bytes is not None:
        reserves.append(int(settings.memory_min_free_bytes))
    if (
        settings.memory_load_headroom_fraction is not None
        and snapshot.system_total_bytes is not None
    ):
        reserves.append(
            int(snapshot.system_total_bytes * settings.memory_load_headroom_fraction)
        )
    return max(reserves, default=0)


def build_model_memory_estimate(
    model_name: str,
    model_type: str,
    model_dir: Path | None = None,
    estimated_weight_bytes: int | None = None,
    extra_required_bytes: int = 0,
    snapshot: MemorySnapshot | None = None,
) -> ModelMemoryEstimate:
    """Build a best-effort memory estimate for a local model load."""
    snapshot = snapshot or collect_memory_snapshot()
    if estimated_weight_bytes is None:
        estimated_weight_bytes = estimate_model_size_bytes(model_dir) if model_dir else 0
    extra_required_bytes = max(0, int(extra_required_bytes))
    return ModelMemoryEstimate(
        model_name=model_name,
        model_type=model_type,
        estimated_weight_bytes=estimated_weight_bytes,
        required_bytes=_estimate_required_bytes(estimated_weight_bytes) + extra_required_bytes,
        system_available_bytes=snapshot.system_available_bytes,
        reserved_headroom_bytes=_reserved_headroom_bytes(snapshot),
    )


def check_model_load_memory(
    model_name: str,
    model_type: str,
    model_dir: Path | None = None,
    estimated_weight_bytes: int | None = None,
    extra_required_bytes: int = 0,
    snapshot: MemorySnapshot | None = None,
) -> ModelMemoryEstimate:
    """Raise when a model load is expected to exceed available memory."""
    estimate = build_model_memory_estimate(
        model_name,
        model_type,
        model_dir,
        estimated_weight_bytes,
        extra_required_bytes,
        snapshot,
    )
    if not settings.memory_load_guard_enabled:
        return estimate

    # Unknown or zero estimates are allowed. The runtime memory guard still
    # protects admissions after the load starts.
    if estimate.required_bytes <= 0 or estimate.available_for_load_bytes is None:
        return estimate

    if estimate.required_bytes > estimate.available_for_load_bytes:
        raise ModelLoadMemoryError(estimate)

    return estimate


def _read_model_config(model_dir: Path | None) -> dict:
    return _read_json_file(model_dir / "config.json" if model_dir else None)


def _read_json_file(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _iter_text_config_candidates(config: dict):
    """Yield likely language-model configs from converted HF/MLX config files."""
    if config:
        yield config
    for key in (
        "text_config",
        "llm_config",
        "language_config",
        "language_model_config",
        "model_config",
        "decoder_config",
    ):
        nested = config.get(key)
        if isinstance(nested, dict):
            yield nested


def _iter_nested_config_candidates(config: dict, *keys: str):
    """Yield a config and selected nested dict configs."""
    if config:
        yield config
    for key in keys:
        nested = config.get(key)
        if isinstance(nested, dict):
            yield nested


def _first_int_from_configs(configs, *keys: str) -> int | None:
    for candidate in configs:
        value = _first_int(candidate, *keys)
        if value is not None:
            return value
    return None


def _coerce_positive_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def _coerce_square_size(value) -> tuple[int, int] | None:
    size = _coerce_positive_int(value)
    if size is not None:
        return size, size
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = _coerce_positive_int(value[0])
        width = _coerce_positive_int(value[1])
        if height and width:
            return height, width
    if not isinstance(value, dict):
        return None

    if "height" in value and "width" in value:
        height = _coerce_positive_int(value.get("height"))
        width = _coerce_positive_int(value.get("width"))
        if height and width:
            return height, width

    square_sizes = [
        size
        for key in ("shortest_edge", "longest_edge", "max_height", "max_width")
        if (size := _coerce_positive_int(value.get(key))) is not None
    ]
    if square_sizes:
        size = max(square_sizes)
        return size, size

    nested = value.get("size")
    if nested is not None:
        return _coerce_square_size(nested)

    return None


def _first_size_from_configs(configs, *keys: str) -> tuple[int, int] | None:
    for candidate in configs:
        for key in keys:
            size = _coerce_square_size(candidate.get(key))
            if size is not None:
                return size
    return None


def _first_scalar_from_configs(configs, *keys: str) -> int | None:
    for candidate in configs:
        for key in keys:
            value = candidate.get(key)
            scalar = _coerce_positive_int(value)
            if scalar is not None:
                return scalar
            if isinstance(value, (list, tuple)) and value:
                scalar = _coerce_positive_int(value[0])
                if scalar is not None:
                    return scalar
    return None


def _iter_vlm_config_candidates(model_dir: Path | None):
    if model_dir is None:
        return
    config = _read_model_config(model_dir)
    processor_config = _read_json_file(model_dir / "processor_config.json")
    preprocessor_config = _read_json_file(model_dir / "preprocessor_config.json")
    for source in (config, processor_config, preprocessor_config):
        if not isinstance(source, dict):
            continue
        yield from _iter_nested_config_candidates(
            source,
            "vision_config",
            "vision_model_config",
            "image_processor",
            "image_processor_config",
            "preprocessor_config",
            "processor_config",
        )


def estimate_vlm_image_tokens(model_dir: Path | None, image_count: int) -> int:
    """Estimate VLM image token reservations from processor/model config."""
    image_count = max(0, int(image_count))
    if image_count <= 0:
        return 0

    fallback = max(0, int(settings.memory_vlm_image_tokens_per_image))
    candidates = list(_iter_vlm_config_candidates(model_dir))
    if not candidates:
        return fallback * image_count

    explicit_tokens = _first_scalar_from_configs(
        candidates,
        "num_image_tokens",
        "image_seq_length",
        "image_token_count",
        "tokens_per_image",
    )
    patch_size = _first_scalar_from_configs(candidates, "patch_size", "image_patch_size")
    merge_size = _first_scalar_from_configs(
        candidates,
        "spatial_merge_size",
        "merge_size",
        "spatial_merge_unit",
    )
    image_size = _first_size_from_configs(
        candidates,
        "crop_size",
        "size",
        "image_size",
        "default_image_size",
        "input_size",
    )
    max_pixels = _first_scalar_from_configs(candidates, "max_pixels")

    tokens_per_image = explicit_tokens or 0
    if patch_size:
        merge_area = max(1, int(merge_size or 1) ** 2)
        if max_pixels:
            tokens_per_image = max(
                tokens_per_image,
                math.ceil(max_pixels / (patch_size * patch_size * merge_area)),
            )
        elif image_size:
            height, width = image_size
            tokens_per_image = max(
                tokens_per_image,
                math.ceil(height / patch_size)
                * math.ceil(width / patch_size)
                // merge_area,
            )

    tokens_per_image = max(fallback, int(tokens_per_image))
    return tokens_per_image * image_count


def _first_int(config: dict, *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _quantized_kv_bytes_per_scalar() -> float | None:
    kv_bits = settings.generation_kv_bits
    if kv_bits is None:
        return None
    group_size = max(1, int(settings.generation_kv_group_size))
    # Quantized MLX caches still carry per-group scale metadata. Model this as
    # quantized payload bits plus one fp16 scale per group.
    return (max(1, int(kv_bits)) / 8.0) + (2.0 / group_size)


def _apply_kv_quantization_estimate(base_bytes_per_token: int, total_tokens: int) -> int:
    """Scale a fp16 KV estimate when generation KV quantization is enabled."""
    quantized_bytes_per_scalar = _quantized_kv_bytes_per_scalar()
    if quantized_bytes_per_scalar is None:
        return base_bytes_per_token * total_tokens

    fp16_bytes_per_scalar = 2.0
    quantized_start = max(0, int(settings.generation_quantized_kv_start))
    fp16_tokens = min(total_tokens, quantized_start)
    quantized_tokens = max(0, total_tokens - fp16_tokens)
    fp16_bytes = base_bytes_per_token * fp16_tokens
    quantized_bytes = (
        base_bytes_per_token
        * quantized_tokens
        * (quantized_bytes_per_scalar / fp16_bytes_per_scalar)
    )
    return int(math.ceil(fp16_bytes + quantized_bytes))


def estimate_generation_kv_cache_bytes(
    model_dir: Path | None,
    total_tokens: int,
) -> int:
    """Estimate KV cache memory from model config and total context tokens."""
    if total_tokens <= 0:
        return 0

    config = _read_model_config(model_dir)
    candidates = list(_iter_text_config_candidates(config))
    layers = _first_int_from_configs(candidates, "num_hidden_layers", "n_layer", "n_layers")
    attention_heads = _first_int_from_configs(
        candidates,
        "num_attention_heads",
        "n_head",
        "n_heads",
    )
    kv_heads = _first_int_from_configs(
        candidates,
        "num_key_value_heads",
        "n_kv_heads",
        "num_kv_heads",
    )
    hidden_size = _first_int_from_configs(candidates, "hidden_size", "n_embd", "d_model")
    head_dim = _first_int_from_configs(candidates, "head_dim")
    kv_lora_rank = _first_int_from_configs(candidates, "kv_lora_rank")
    qk_rope_head_dim = _first_int_from_configs(candidates, "qk_rope_head_dim")

    if head_dim is None and hidden_size is not None and attention_heads:
        head_dim = hidden_size // attention_heads
    if kv_heads is None:
        kv_heads = attention_heads

    base_bytes_per_token = 0
    if layers and kv_heads and head_dim:
        bytes_per_scalar = 2
        key_and_value = 2
        base_bytes_per_token = layers * kv_heads * head_dim * key_and_value * bytes_per_scalar

    # MLA-style caches, such as DeepSeek variants, store compressed KV plus
    # positional key state rather than full key/value heads.
    if layers and (kv_lora_rank or qk_rope_head_dim):
        bytes_per_scalar = 2
        mla_scalars = int(kv_lora_rank or 0) + int(qk_rope_head_dim or 0)
        base_bytes_per_token = max(base_bytes_per_token, layers * mla_scalars * bytes_per_scalar)

    if base_bytes_per_token > 0:
        return _apply_kv_quantization_estimate(base_bytes_per_token, total_tokens)

    fallback = settings.memory_generation_kv_bytes_per_token
    if fallback <= 0:
        return 0
    return _apply_kv_quantization_estimate(fallback, total_tokens)


def build_generation_memory_estimate(
    model_name: str,
    model_type: str,
    prompt_tokens: int,
    max_tokens: int,
    image_tokens: int = 0,
    model_dir: Path | None = None,
    snapshot: MemorySnapshot | None = None,
) -> GenerationMemoryEstimate:
    """Build a best-effort generation KV cache memory estimate."""
    snapshot = snapshot or collect_memory_snapshot()
    prompt_tokens = max(0, int(prompt_tokens))
    max_tokens = max(0, int(max_tokens))
    image_tokens = max(0, int(image_tokens))
    kv_cache_bytes = estimate_generation_kv_cache_bytes(
        model_dir,
        prompt_tokens + max_tokens + image_tokens,
    )
    return GenerationMemoryEstimate(
        model_name=model_name,
        model_type=model_type,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        kv_cache_bytes=kv_cache_bytes,
        system_available_bytes=snapshot.system_available_bytes,
        reserved_headroom_bytes=_reserved_headroom_bytes(snapshot),
        image_tokens=image_tokens,
    )


def check_generation_memory(
    model_name: str,
    model_type: str,
    prompt_tokens: int,
    max_tokens: int,
    image_tokens: int = 0,
    model_dir: Path | None = None,
    snapshot: MemorySnapshot | None = None,
) -> GenerationMemoryEstimate:
    """Raise when generation KV cache memory is expected to exceed available memory."""
    estimate = build_generation_memory_estimate(
        model_name,
        model_type,
        prompt_tokens,
        max_tokens,
        image_tokens,
        model_dir,
        snapshot,
    )
    if not settings.memory_generation_guard_enabled:
        return estimate

    if estimate.kv_cache_bytes <= 0 or estimate.available_for_request_bytes is None:
        return estimate

    if estimate.kv_cache_bytes > estimate.available_for_request_bytes:
        raise GenerationMemoryError(estimate)

    return estimate
