"""Embeddings API router - OpenAI compatible."""

import asyncio
import base64
import logging
import threading
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.config import settings
from mlx_serve.core.batch_processor import EmbeddingBatchProcessor
from mlx_serve.core.inference_control import (
    InferenceOverloadedError,
    build_inference_key,
    build_overload_detail,
    get_model_execution_lock,
    raise_if_server_overloaded,
)
from mlx_serve.core.mlx_memory import clear_mlx_cache
from mlx_serve.core.model_manager import (
    ModelType,
    model_manager,
    register_model_unload_hook,
    resolve_model_alias,
)
from mlx_serve.core.model_memory import ModelLoadMemoryError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

# Cache for batch processors per model
_batch_processors: dict[str, EmbeddingBatchProcessor] = {}
_batch_processors_lock = threading.Lock()


def release_embedding_batch_processors(model_name: str | None = None) -> int:
    """Stop and remove embedding batch processors for one model, or all models."""
    if model_name is None:
        processor_keys: list[str] | None = None
    else:
        processor_keys = [build_inference_key("embedding", model_name)]

    with _batch_processors_lock:
        if processor_keys is None:
            processors = list(_batch_processors.items())
            _batch_processors.clear()
        else:
            processors = [
                (key, _batch_processors.pop(key))
                for key in processor_keys
                if key in _batch_processors
            ]

    for _key, processor in processors:
        close_nowait = getattr(processor, "close_nowait", None)
        if callable(close_nowait):
            close_nowait()
            continue
        stop_nowait = getattr(processor, "stop_nowait", None)
        if callable(stop_nowait):
            stop_nowait()

    if processors:
        logger.info("Released %s embedding batch processor(s)", len(processors))
    return len(processors)


def _release_processors_for_removed_model(
    model_type: ModelType,
    model_name: str,
    _reason: str,
) -> None:
    if model_type == "embedding":
        release_embedding_batch_processors(model_name)


register_model_unload_hook(_release_processors_for_removed_model)


def truncate_embedding(embedding: list[float], dimensions: int) -> list[float]:
    """Truncate embedding to specified dimensions with L2 normalization.

    This implements the Matryoshka Representation Learning approach where
    embeddings can be shortened while preserving semantic information.

    Args:
        embedding: Original embedding vector.
        dimensions: Target number of dimensions.

    Returns:
        Truncated and L2-normalized embedding.
    """
    truncated = np.array(embedding[:dimensions], dtype=np.float32)

    # L2 normalization (vector magnitude = 1)
    norm = np.linalg.norm(truncated)
    if norm > 0:
        truncated = truncated / norm

    return truncated.tolist()


def encode_embedding_base64(embedding: list[float]) -> str:
    """Encode embedding as base64 string (OpenAI compatible format).

    Uses float32 little-endian format as per OpenAI API specification.

    Args:
        embedding: Embedding vector as list of floats.

    Returns:
        Base64-encoded string.
    """
    # float32, little-endian (OpenAI compatible)
    embedding_bytes = np.array(embedding, dtype="<f4").tobytes()
    return base64.b64encode(embedding_bytes).decode("utf-8")


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str = Field(..., description="Model name to use for embedding")
    input: str | list[str] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the embedding output",
    )
    dimensions: int | None = Field(
        default=None,
        description="Number of dimensions for the output embedding (requires MRL-trained model)",
        gt=0,
    )


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str  # float array or base64 string
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


async def _generate_embeddings_batch(
    model, tokenizer, texts: list[str]
) -> list[list[float]]:
    """Generate embeddings using batch processing."""
    import mlx.core as mx

    loop = asyncio.get_running_loop()

    def _generate():
        # Get the underlying tokenizer if wrapped (TokenizerWrapper doesn't support __call__)
        tok = getattr(tokenizer, "_tokenizer", tokenizer)

        # Tokenize texts - use __call__ instead of batch_encode_plus for compatibility
        # with tokenizers that don't support batch_encode_plus (e.g., Qwen2Tokenizer)
        inputs = tok(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Convert to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Generate embeddings
        result = model(input_ids, attention_mask=attention_mask)
        return result.text_embeds.tolist()

    return await loop.run_in_executor(None, _generate)


async def _embed_texts(
    model_name: str,
    model,
    tokenizer,
    texts: list[str],
) -> list[list[float]]:
    """Route requests through the shared embedding batch processor."""
    processor_key = build_inference_key("embedding", model_name)
    old_processor: EmbeddingBatchProcessor | None = None

    with _batch_processors_lock:
        processor = _batch_processors.get(processor_key)
        if (
            processor is None
            or processor.model is not model
            or processor.tokenizer is not tokenizer
        ):
            old_processor = processor
            processor = EmbeddingBatchProcessor(
                model,
                tokenizer,
                execution_lock=get_model_execution_lock(processor_key),
            )
            _batch_processors[processor_key] = processor

    if old_processor is not None:
        old_processor.close_nowait()

    return await processor.embed(texts)


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the given input(s).

    Supports batch processing for improved throughput.
    """
    # Normalize input to list
    texts = request.input if isinstance(request.input, list) else [request.input]
    canonical_model_name, _, _ = resolve_model_alias(request.model)

    if not texts:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Input cannot be empty",
                    "type": "invalid_request_error",
                    "code": "invalid_input",
                }
            },
        )

    try:
        raise_if_server_overloaded()
        model, tokenizer = model_manager.get_embedding_model(request.model)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        ) from e
    except ModelLoadMemoryError as e:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(str(e)),
        ) from e
    except InferenceOverloadedError as e:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(
                f"Embedding model '{request.model}' is overloaded. {e}"
            ),
        ) from e

    try:
        # Generate embeddings with batch processing
        embeddings_list = await _embed_texts(canonical_model_name, model, tokenizer, texts)
        # Approximate token count from the returned embeddings count; avoids a
        # redundant tokenizer pass that the batch processor already performed.
        total_tokens = sum(len(text.split()) for text in texts)

        # Post-process embeddings: dimensions truncation + encoding format
        processed_embeddings: list[list[float] | str] = []
        for emb in embeddings_list:
            # 1. Apply dimensions truncation if requested
            if request.dimensions:
                if request.dimensions > len(emb):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "Requested dimensions "
                                    f"({request.dimensions}) exceeds model "
                                    f"embedding size ({len(emb)})"
                                ),
                                "type": "invalid_request_error",
                                "code": "invalid_dimensions",
                            }
                        },
                    )
                if request.dimensions < len(emb):
                    emb = truncate_embedding(emb, request.dimensions)

            # 2. Apply encoding format
            if request.encoding_format == "base64":
                processed_embeddings.append(encode_embedding_base64(emb))
            else:
                processed_embeddings.append(emb)

        # Build response
        data = [
            EmbeddingData(embedding=emb, index=idx)
            for idx, emb in enumerate(processed_embeddings)
        ]

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    except InferenceOverloadedError as e:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(
                f"Embedding model '{request.model}' is overloaded. {e}"
            ),
        ) from e
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Embedding generation failed: {str(e)}",
                    "type": "server_error",
                    "code": "embedding_failed",
                }
            },
        ) from e
    finally:
        if settings.retrieval_clear_mlx_cache_after_request:
            clear_mlx_cache(log=logger, reason="/v1/embeddings")
