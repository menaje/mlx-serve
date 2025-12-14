"""Reranking API router - Jina compatible extension."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rerank"])


class RerankRequest(BaseModel):
    """Jina-compatible rerank request."""

    model: str = Field(..., description="Model name to use for reranking")
    query: str = Field(..., description="Search query")
    documents: list[str] = Field(..., description="Documents to rerank")
    top_n: int | None = Field(default=None, description="Number of top results to return")
    return_documents: bool = Field(
        default=True,
        description="Whether to return document text in results",
    )


class DocumentResult(BaseModel):
    """Document text wrapper."""

    text: str


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int
    relevance_score: float
    document: DocumentResult | None = None


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
) -> float:
    """Compute relevance score between query and document using the reranker model."""
    import mlx.core as mx

    # Format input for cross-encoder reranking
    # Most rerankers expect: [CLS] query [SEP] document [SEP]
    prompt = f"Query: {query}\nDocument: {document}\nRelevant:"

    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="mlx")

    # Get model output
    outputs = model(tokens)

    # Extract logits for relevance scoring
    # For rerankers, we typically look at the logits for "yes"/"no" or similar
    # This is a simplified scoring - actual implementation depends on model architecture
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs

    # Get the last token's logits and convert to probability
    last_logits = logits[0, -1, :]

    # Simple scoring: use the mean of positive logits
    score = float(mx.sigmoid(mx.mean(last_logits)))

    return score


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """Rerank documents based on relevance to the query."""
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
        # Compute scores for all documents
        scored_docs = []
        total_tokens = 0

        for idx, doc in enumerate(request.documents):
            score = compute_rerank_score(model, tokenizer, request.query, doc)
            scored_docs.append((idx, score, doc))

            # Approximate token count
            total_tokens += len(tokenizer.encode(f"{request.query} {doc}"))

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
