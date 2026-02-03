"""Reranking API router - Jina compatible extension."""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rerank"])


class RerankRequest(BaseModel):
    """Jina-compatible rerank request with extended options."""

    model: str = Field(..., description="Model name to use for reranking")
    query: str = Field(..., description="Search query")
    documents: list[str] = Field(..., description="Documents to rerank")
    top_n: int | None = Field(default=None, description="Number of top results to return")
    return_documents: bool = Field(
        default=True,
        description="Whether to return document text in results",
    )
    return_text: bool = Field(
        default=False,
        description="Whether to return yes/no text output based on relevance",
    )
    decision_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for yes/no decision (0.0-1.0)",
    )


class DocumentResult(BaseModel):
    """Document text wrapper."""

    text: str


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int
    relevance_score: float
    document: DocumentResult | None = None
    text_output: str | None = None


class RerankUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    total_tokens: int


class RerankResponse(BaseModel):
    """Jina-compatible rerank response."""

    results: list[RerankResult]
    usage: RerankUsage


def compute_rerank_score(
    model,
    tokenizer,
    query: str,
    document: str,
    instruction: str | None = None,
) -> float:
    """Compute relevance score between query and document using Qwen3-Reranker."""
    import mlx.core as mx
    import mlx.nn as nn

    # Default instruction for retrieval tasks
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"

    # Qwen3-Reranker format
    prompt = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    # Get token IDs for "yes" and "no"
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    # Tokenize with prefix/suffix tokens for Qwen3
    prefix_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    suffix_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = prefix_tokens + input_ids + suffix_tokens

    # Convert to MLX array
    tokens = mx.array([input_ids])

    # Get model output
    outputs = model(tokens)

    # Extract logits
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs

    # Get the last token's logits
    last_logits = logits[0, -1, :]

    # Get scores for "yes" and "no" tokens
    true_score = last_logits[token_true_id]
    false_score = last_logits[token_false_id]

    # Compute probability using softmax over [false, true]
    scores = mx.stack([false_score, true_score])
    probs = mx.softmax(scores)

    # Return probability of "yes"
    return float(probs[1])


async def _compute_batch_scores(
    model, tokenizer, query: str, documents: list[str], instruction: str | None = None
) -> tuple[list[float], int]:
    """Compute rerank scores for all documents using batch processing."""
    loop = asyncio.get_running_loop()

    def _compute():
        import mlx.core as mx

        if instruction is None:
            inst = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            inst = instruction

        # Get token IDs for "yes" and "no"
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        token_false_id = tokenizer.convert_tokens_to_ids("no")

        scores = []
        total_tokens = 0

        # Process all documents
        for doc in documents:
            prompt = f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"

            # Tokenize
            prefix_tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
            suffix_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = prefix_tokens + input_ids + suffix_tokens

            tokens = mx.array([input_ids])
            total_tokens += len(input_ids)

            # Get model output
            outputs = model(tokens)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Get last token logits
            last_logits = logits[0, -1, :]

            # Compute probability
            true_score = last_logits[token_true_id]
            false_score = last_logits[token_false_id]
            probs = mx.softmax(mx.stack([false_score, true_score]))

            scores.append(float(probs[1]))

        return scores, total_tokens

    return await loop.run_in_executor(None, _compute)


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """Rerank documents based on relevance to the query.

    Uses batch processing for improved throughput.
    """
    if not request.documents:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Documents list cannot be empty",
                    "type": "invalid_request_error",
                    "code": "invalid_input",
                }
            },
        )

    try:
        model, tokenizer = model_manager.get_reranker_model(request.model)
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
        # Compute scores for all documents using batch processing
        scores, total_tokens = await _compute_batch_scores(
            model, tokenizer, request.query, request.documents
        )

        # Create scored docs list
        scored_docs = [
            (idx, score, doc)
            for idx, (score, doc) in enumerate(zip(scores, request.documents))
        ]

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit
        if request.top_n is not None:
            scored_docs = scored_docs[: request.top_n]

        # Build results
        results = [
            RerankResult(
                index=idx,
                relevance_score=score,
                document=DocumentResult(text=doc) if request.return_documents else None,
                text_output=("yes" if score >= request.decision_threshold else "no")
                if request.return_text
                else None,
            )
            for idx, score, doc in scored_docs
        ]

        return RerankResponse(
            results=results,
            usage=RerankUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Reranking failed: {str(e)}",
                    "type": "server_error",
                    "code": "rerank_failed",
                }
            },
        ) from e
