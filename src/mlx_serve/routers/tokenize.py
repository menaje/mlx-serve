"""Tokenize API router for token counting."""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from mlx_serve.core.inference_control import (
    InferenceOverloadedError,
    build_inference_key,
    build_overload_detail,
    get_model_execution_lock,
    raise_if_server_overloaded,
)
from mlx_serve.core.retrieval_model_routing import resolve_retrieval_model_type
from mlx_serve.core.runtime_topology import (
    get_retrieval_worker_kind,
    retrieval_worker_isolation_enabled,
)
from mlx_serve.core.model_manager import model_manager
from mlx_serve.routers.retrieval_proxy import forward_to_retrieval_worker

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


def _build_model_not_found_detail(model_name: str) -> dict:
    """Build a consistent model-not-found error payload."""
    return {
        "error": {
            "message": f"Model '{model_name}' not found",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    }


def _load_tokenizer_for_model(
    model_name: str,
    allowed_kind: Literal["embedding", "reranker"] | None = None,
) -> tuple[str, object]:
    """Load the tokenizer for a retrieval model."""
    resolved_kind = resolve_retrieval_model_type(model_name)
    if allowed_kind is not None:
        if resolved_kind is not None and resolved_kind != allowed_kind:
            raise ValueError(f"Model '{model_name}' does not belong to {allowed_kind}")
        if allowed_kind == "embedding":
            _, tokenizer = model_manager.get_embedding_model(model_name)
            return allowed_kind, tokenizer
        _, tokenizer = model_manager.get_reranker_model(model_name)
        return allowed_kind, tokenizer

    if resolved_kind == "embedding":
        _, tokenizer = model_manager.get_embedding_model(model_name)
        return resolved_kind, tokenizer
    if resolved_kind == "reranker":
        _, tokenizer = model_manager.get_reranker_model(model_name)
        return resolved_kind, tokenizer

    # Fallback for legacy local mode if the type cannot be inferred from aliases/metadata.
    try:
        _, tokenizer = model_manager.get_embedding_model(model_name)
        return "embedding", tokenizer
    except ValueError:
        _, tokenizer = model_manager.get_reranker_model(model_name)
        return "reranker", tokenizer


async def _forward_tokenize_request_to_worker(
    http_request: Request,
    worker_kind: Literal["embedding", "reranker"],
    body: bytes,
) -> Response:
    """Forward tokenize traffic to a single retrieval worker."""
    return await forward_to_retrieval_worker(http_request, worker_kind, body=body)


@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest, http_request: Request) -> TokenizeResponse | Response:
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

    if retrieval_worker_isolation_enabled():
        body = request.model_dump_json().encode("utf-8")
        resolved_kind = resolve_retrieval_model_type(request.model)
        if resolved_kind is None:
            candidate_kinds: list[Literal["embedding", "reranker"]] = [
                "embedding",
                "reranker",
            ]
        else:
            candidate_kinds = [resolved_kind]
            fallback_kind = "reranker" if resolved_kind == "embedding" else "embedding"
            candidate_kinds.append(fallback_kind)

        last_response: Response | None = None
        for worker_kind in candidate_kinds:
            response = await _forward_tokenize_request_to_worker(
                http_request,
                worker_kind,
                body,
            )
            if response.status_code != 404:
                return response
            last_response = response

        if last_response is not None:
            return last_response
        raise HTTPException(
            status_code=404,
            detail=_build_model_not_found_detail(request.model),
        )

    worker_kind = get_retrieval_worker_kind()
    allowed_kind: Literal["embedding", "reranker"] | None = worker_kind

    try:
        raise_if_server_overloaded()
        model_type, tokenizer = _load_tokenizer_for_model(
            request.model,
            allowed_kind=allowed_kind,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=_build_model_not_found_detail(request.model),
        )

    try:
        model_key = build_inference_key(model_type, request.model)
        async with get_model_execution_lock(model_key):
            data = await _tokenize_texts(tokenizer, texts, request.return_tokens)

        return TokenizeResponse(
            data=data,
            model=request.model,
        )

    except InferenceOverloadedError as e:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(
                f"Tokenizer for model '{request.model}' is overloaded. {e}"
            ),
        ) from e
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
