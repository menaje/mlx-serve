"""Gateway proxy routes for isolated LLM/VLM workers."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_serve.core.inference_control import build_overload_detail
from mlx_serve.core.model_manager import model_manager, resolve_model_alias
from mlx_serve.core.runtime_topology import GenerationWorkerKind
from mlx_serve.routers.retrieval_proxy import perform_worker_request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generation"])


@dataclass
class WorkerStreamingResponse:
    """Open worker response handle for streaming proxying."""

    status_code: int
    headers: dict[str, str]
    response: object


def _content_has_image(content: object) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("image_url"):
            return True
    return False


def resolve_generation_worker_kind(body: dict) -> GenerationWorkerKind:
    """Resolve a chat/completion request body to the worker that should own it."""
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and _content_has_image(message.get("content")):
                return "vlm"

    model_name = str(body.get("model") or "")
    resolved_name, _, resolved_type = resolve_model_alias(model_name)
    if resolved_type == "vlm":
        return "vlm"
    if resolved_type == "llm":
        return "llm"

    installed_type = model_manager.get_model_type(model_name)
    if installed_type == "vlm":
        return "vlm"
    if installed_type == "llm":
        return "llm"

    if resolved_name != model_name:
        installed_type = model_manager.get_model_type(resolved_name)
        if installed_type == "vlm":
            return "vlm"

    return "llm"


def open_worker_stream(
    url: str,
    method: str,
    body: bytes,
    headers: dict[str, str],
) -> WorkerStreamingResponse | Response:
    """Open a streaming worker request without reading the full body."""
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        return Response(
            content=exc.read(),
            status_code=exc.code,
            media_type=exc.headers.get("Content-Type", "application/json"),
        )
    return WorkerStreamingResponse(
        status_code=response.status,
        headers=dict(response.headers.items()),
        response=response,
    )


def _iter_worker_response(response: object):
    try:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            yield chunk
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


async def forward_to_generation_worker(
    request: Request,
    worker_kind: GenerationWorkerKind,
    model_name: str,
    body: bytes | None = None,
    stream: bool = False,
) -> Response:
    """Forward a request payload to the selected generation worker."""
    worker_urls = getattr(request.app.state, "generation_worker_urls", {})
    base_url = worker_urls.get(worker_kind)
    supervisor = getattr(request.app.state, "generation_worker_supervisor", None)
    if supervisor is not None and hasattr(supervisor, "ensure_worker"):
        base_url = supervisor.ensure_worker(worker_kind, model_name)
    if base_url is None:
        raise HTTPException(
            status_code=503,
            detail=build_overload_detail(f"{worker_kind} generation worker is unavailable"),
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

    if stream:
        try:
            opened_response = await asyncio.to_thread(
                open_worker_stream,
                url,
                request.method,
                body,
                headers,
            )
        except urllib.error.URLError as exc:
            logger.error("Generation worker stream failed: %s", exc)
            return JSONResponse(
                status_code=503,
                content=build_overload_detail(f"{worker_kind} generation worker is unavailable"),
            )
        if isinstance(opened_response, Response):
            return opened_response
        media_type = opened_response.headers.get("Content-Type", "text/event-stream")
        return StreamingResponse(
            _iter_worker_response(opened_response.response),
            status_code=opened_response.status_code,
            media_type=media_type,
        )

    try:
        response = await asyncio.to_thread(
            perform_worker_request,
            url,
            request.method,
            body,
            headers,
        )
    except urllib.error.URLError as exc:
        logger.error("Generation worker request failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content=build_overload_detail(f"{worker_kind} generation worker is unavailable"),
        )

    media_type = response.headers.get("Content-Type", "application/json")
    return Response(
        content=response.body,
        status_code=response.status_code,
        media_type=media_type,
    )


async def _request_json_body(request: Request) -> tuple[bytes, dict]:
    body = await request.body()
    try:
        parsed = json.loads(body.decode("utf-8")) if body else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Request body must be valid JSON",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Request body must be a JSON object",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
        )
    return body, parsed


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """Forward chat completions to an isolated LLM or VLM worker."""
    body, parsed = await _request_json_body(request)
    return await forward_to_generation_worker(
        request,
        resolve_generation_worker_kind(parsed),
        str(parsed.get("model") or ""),
        body=body,
        stream=bool(parsed.get("stream")),
    )


@router.post("/v1/completions")
async def proxy_text_completions(request: Request) -> Response:
    """Forward legacy text completions to the isolated LLM worker."""
    body, parsed = await _request_json_body(request)
    return await forward_to_generation_worker(
        request,
        "llm",
        str(parsed.get("model") or ""),
        body=body,
        stream=bool(parsed.get("stream")),
    )
