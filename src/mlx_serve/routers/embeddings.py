"""Embeddings API router - OpenAI compatible."""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

# Cache for batch processors per model
_batch_processors: dict = {}


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str = Field(..., description="Model name to use for embedding")
    input: str | list[str] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the embedding output",
    )


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
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

    loop = asyncio.get_event_loop()

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
        loop = asyncio.get_event_loop()
        total_tokens = await loop.run_in_executor(
            None,
            lambda: sum(len(tokenizer.encode(text)) for text in texts)
        )

        # Build response
        data = [
            EmbeddingData(embedding=emb, index=idx)
            for idx, emb in enumerate(embeddings_list)
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
