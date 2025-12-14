"""Model quantization utilities for mlx-serve."""

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

from mlx_serve.config import settings

logger = logging.getLogger(__name__)

QuantizationBits = Literal[4, 8]


def get_quantized_model_name(model_name: str, bits: QuantizationBits) -> str:
    """Generate quantized model name.

    Args:
        model_name: Original model name.
        bits: Quantization bits.

    Returns:
        Quantized model name (e.g., "model-4bit").
    """
    return f"{model_name}-{bits}bit"


def quantize_model(
    model_name: str,
    bits: QuantizationBits = 4,
    group_size: int = 64,
) -> tuple[bool, str]:
    """
    Quantize a model to reduce memory usage.

    Uses mlx-lm's quantization capabilities.

    Args:
        model_name: Name of the model to quantize.
        bits: Number of bits for quantization (4 or 8).
        group_size: Group size for quantization.

    Returns:
        Tuple of (success, message).
    """
    try:
        from mlx_lm import convert

        # Get model path
        safe_name = model_name.replace("/", "--")
        model_dir = settings.models_dir / safe_name

        if not model_dir.exists():
            return False, f"Model '{model_name}' not found at {model_dir}"

        # Create output directory for quantized model
        quantized_name = get_quantized_model_name(model_name, bits)
        safe_quantized_name = quantized_name.replace("/", "--")
        output_dir = settings.models_dir / safe_quantized_name

        if output_dir.exists():
            return False, f"Quantized model already exists at {output_dir}"

        logger.info(f"Quantizing {model_name} to {bits}-bit...")

        # Use mlx-lm's convert function for quantization
        convert(
            str(model_dir),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            mlx_path=str(output_dir),
        )

        # Update metadata
        metadata_path = settings.models_dir.parent / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = {}

        # Copy original metadata and update
        original_meta = metadata.get(model_name, {})
        metadata[quantized_name] = {
            "type": original_meta.get("type", "embedding"),
            "hf_repo": original_meta.get("hf_repo", ""),
            "quantization": {
                "bits": bits,
                "group_size": group_size,
            },
            "original_model": model_name,
            "size": _get_dir_size(output_dir),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Quantization complete: {quantized_name}")
        return True, f"Successfully quantized to {quantized_name}"

    except ImportError:
        return False, "mlx-lm is required for quantization. Install with: pip install mlx-lm"
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False, f"Quantization failed: {str(e)}"


def _get_dir_size(path: Path) -> int:
    """Calculate total size of a directory."""
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def get_quantization_info(model_name: str) -> dict | None:
    """
    Get quantization info for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Quantization info dict or None if not quantized.
    """
    metadata_path = settings.models_dir.parent / "metadata.json"
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    model_meta = metadata.get(model_name, {})

    return model_meta.get("quantization")


def is_quantized_model(model_name: str) -> bool:
    """Check if a model is quantized."""
    return get_quantization_info(model_name) is not None


def list_quantization_options() -> list[dict]:
    """List available quantization options."""
    return [
        {
            "bits": 4,
            "description": "4-bit quantization - Best compression, slight accuracy loss",
            "memory_reduction": "~75%",
        },
        {
            "bits": 8,
            "description": "8-bit quantization - Good compression, minimal accuracy loss",
            "memory_reduction": "~50%",
        },
    ]
