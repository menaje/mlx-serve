"""Tokenize API router for token counting."""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tokenize"])


class TokenizeRequest(BaseModel):
    """Token count request."""

    model: str = Field(..., description="Model name to use for tokenization")
    input: str | list[str] = Field(..., description="Text(s) to tokenize")
    return_tokens: bool = Field(
        default=False,
        description="Whether to return token IDs in the response",
    )


class TokenData(BaseModel):
    """Single tokenization result."""

    index: int
    tokens: int
    token_ids: list[int] | None = None


class TokenizeResponse(BaseModel):
    """Token count response."""

    object: Literal["list"] = "list"
    data: list[TokenData]
    model: str


async def _tokenize_texts(
    tokenizer, texts: list[str], return_tokens: bool
) -> list[TokenData]:
    """Tokenize texts and return token counts."""
    loop = asyncio.get_running_loop()

    def _tokenize():
        results = []
        for idx, text in enumerate(texts):
            token_ids = tokenizer.encode(text)
            results.append(
                TokenData(
                    index=idx,
                    tokens=len(token_ids),
                    token_ids=token_ids if return_tokens else None,
                )
            )
        return results

    return await loop.run_in_executor(None, _tokenize)


@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest) -> TokenizeResponse:
    """Count tokens for the given input(s).

    Returns the number of tokens for each input text.
    Optionally returns the token IDs as well.
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

    # Try to get tokenizer from embedding model first, then reranker
    tokenizer = None
    try:
        _, tokenizer = model_manager.get_embedding_model(request.model)
    except ValueError:
        try:
            _, tokenizer = model_manager.get_reranker_model(request.model)
        except ValueError:
            pass

    if tokenizer is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    try:
        data = await _tokenize_texts(tokenizer, texts, request.return_tokens)

        return TokenizeResponse(
            data=data,
            model=request.model,
        )

    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Tokenization failed: {str(e)}",
                    "type": "server_error",
                    "code": "tokenize_failed",
                }
            },
        ) from e
