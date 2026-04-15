"""Gateway proxy routes for isolated retrieval workers."""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from mlx_serve.core.inference_control import build_overload_detail
from mlx_serve.core.runtime_topology import RetrievalWorkerKind

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])


@dataclass
class WorkerProxyResponse:
    """Response payload returned by an internal worker."""

    status_code: int
    body: bytes
    headers: dict[str, str]


def perform_worker_request(
    url: str,
    method: str,
    body: bytes,
    headers: dict[str, str],
) -> WorkerProxyResponse:
    """Perform a blocking HTTP request against an internal worker."""
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request) as response:
            return WorkerProxyResponse(
                status_code=response.status,
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        return WorkerProxyResponse(
            status_code=exc.code,
            body=exc.read(),
            headers=dict(exc.headers.items()),
        )


async def forward_to_retrieval_worker(
    request: Request,
    worker_kind: RetrievalWorkerKind,
    body: bytes | None = None,
) -> Response:
    """Forward a request payload to the selected retrieval worker."""
    worker_urls = getattr(request.app.state, "retrieval_worker_urls", {})
    base_url = worker_urls.get(worker_kind)
    if base_url is None:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(f"{worker_kind} retrieval worker is unavailable"),
        )

    path = request.url.path
    query = request.url.query
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"

    if body is None:
        body = await request.body()
    headers: dict[str, str] = {}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type

    try:
        response = await asyncio.to_thread(
            perform_worker_request,
            url,
            request.method,
            body,
            headers,
        )
    except urllib.error.URLError as exc:
        logger.error("Retrieval worker request failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content=build_overload_detail(f"{worker_kind} retrieval worker is unavailable"),
        )

    media_type = response.headers.get("Content-Type", "application/json")
    return Response(
        content=response.body,
        status_code=response.status_code,
        media_type=media_type,
    )


@router.post("/v1/embeddings")
async def proxy_embeddings(request: Request) -> Response:
    """Forward embedding requests to the isolated embedding worker."""
    return await forward_to_retrieval_worker(request, "embedding")


@router.post("/v1/rerank")
async def proxy_rerank(request: Request) -> Response:
    """Forward rerank requests to the isolated reranker worker."""
    return await forward_to_retrieval_worker(request, "reranker")
