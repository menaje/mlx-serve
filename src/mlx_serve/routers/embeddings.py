"""Embeddings API router - OpenAI compatible."""

import asyncio
import base64
import logging
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

# Cache for batch processors per model
_batch_processors: dict = {}


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
    from mlx_embeddings import generate

    loop = asyncio.get_running_loop()

    def _generate():
        result = generate(model, tokenizer, texts)
        return result.text_embeds.tolist()

    return await loop.run_in_executor(None, _generate)


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the given input(s).

    Supports batch processing for improved throughput.
    """
    # Normalize input to list
    texts = request.input if isinstance(request.input, list) else [request.input]

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

    try:
        # Generate embeddings with batch processing
        embeddings_list = await _generate_embeddings_batch(model, tokenizer, texts)

        # Calculate token count (approximate)
        loop = asyncio.get_running_loop()
        total_tokens = await loop.run_in_executor(
            None,
            lambda: sum(len(tokenizer.encode(text)) for text in texts)
        )

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
                                "message": f"Requested dimensions ({request.dimensions}) exceeds model embedding size ({len(emb)})",
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
