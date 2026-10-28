"""Models API router - OpenAI and Ollama compatible."""

import asyncio
import json
import logging
import urllib.error
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import (
    ModelInUseError,
    ModelType,
    model_manager,
    resolve_model_alias,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


async def _decision_worker_request(
    request: Request, path: str, payload: dict | None = None
) -> dict | None:
    """Contact the isolated decision worker, if this gateway has one."""
    from mlx_serve.routers.retrieval_proxy import perform_worker_request

    base_url = getattr(request.app.state, "retrieval_worker_urls", {}).get("decision")
    if not base_url:
        return None
    try:
        response = await asyncio.to_thread(
            perform_worker_request,
            base_url + path,
            "GET" if payload is None else "POST",
            b"" if payload is None else json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
    except (TimeoutError, urllib.error.URLError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {"message": "Decision worker is unavailable", "code": "worker_unavailable"}
            },
        ) from exc
    data = json.loads(response.body)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=data.get("detail", data))
    return data


def _merge_decision_cache(local: dict, worker: dict | None) -> dict:
    if worker is not None:
        local["decision_models"] = worker["decision_models"]
        local["decision_worker_memory"] = worker["mlx_memory"]
        local["decision_worker_active_inference"] = worker["active_inference"]
    return local


# OpenAI-compatible schemas
class ModelData(BaseModel):
    """OpenAI-compatible model info."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "mlx-serve"
    type: ModelType = "embedding"


class ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""

    object: Literal["list"] = "list"
    data: list[ModelData]


# Ollama-compatible schemas
class OllamaModelInfo(BaseModel):
    """Ollama-compatible model info."""

    name: str
    size: int
    type: ModelType
    modified_at: str


class OllamaTagsResponse(BaseModel):
    """Ollama-compatible tags response."""

    models: list[OllamaModelInfo]


class PullRequest(BaseModel):
    """Model pull request."""

    name: str = Field(..., description="Hugging Face repo ID (e.g., Qwen/Qwen3-Embedding-0.6B)")
    type: ModelType = Field(default="embedding", description="Model type")
    revision: str | None = Field(default=None, description="Hugging Face branch, tag or commit")
    model_name: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", description="Local model name"
    )
    quantize: int | None = Field(
        default=None, description="Quantization bits (4 or 8). None for no quantization."
    )
    keep_original: bool = Field(
        default=False, description="Keep original model in HuggingFace cache after conversion"
    )


class DeleteRequest(BaseModel):
    """Model delete request."""

    name: str = Field(..., description="Model name to delete")


class ShowRequest(BaseModel):
    """Model show request."""

    name: str = Field(..., description="Model name to show")


class UnloadRequest(BaseModel):
    """Model unload request."""

    all: bool = Field(default=False, description="Unload all cached idle models")
    type: ModelType | None = Field(default=None, description="Optional model type filter")


class ShowResponse(BaseModel):
    """Model show response."""

    name: str
    type: ModelType
    size: int
    modified_at: str
    hf_repo: str
    path: str


class UnloadResponse(BaseModel):
    """Model unload response."""

    status: Literal["success"] = "success"
    unloaded: list[dict[str, str]]
    skipped_active: list[dict[str, str]] = Field(default_factory=list)
    cache: dict | None = None


# OpenAI-compatible endpoints
@router.get("/v1/models", response_model=ModelsResponse)
async def list_models_openai() -> ModelsResponse:
    """List all available models (OpenAI format)."""
    models = model_manager.list_models()

    return ModelsResponse(
        data=[
            ModelData(
                id=m.name,
                type=m.model_type,
            )
            for m in models
        ]
    )


# Ollama-compatible endpoints
@router.get("/api/tags", response_model=OllamaTagsResponse)
async def list_models_ollama() -> OllamaTagsResponse:
    """List all available models (Ollama format)."""
    models = model_manager.list_models()

    return OllamaTagsResponse(
        models=[
            OllamaModelInfo(
                name=m.name,
                size=m.size,
                type=m.model_type,
                modified_at=m.modified_at,
            )
            for m in models
        ]
    )


@router.post("/api/pull")
async def pull_model(request: PullRequest) -> StreamingResponse:
    """Download and convert a model from Hugging Face to MLX format.

    Supports optional quantization (4-bit or 8-bit) during conversion.
    The d1 preparation path does not delete shared Hugging Face caches.
    """
    import json

    async def generate():
        async for status in model_manager.pull_model(
            hf_repo=request.name,
            model_type=request.type,
            model_name=request.model_name,
            quantize=request.quantize,
            keep_original=request.keep_original,
            revision=request.revision,
        ):
            yield json.dumps(status) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@router.delete("/api/delete")
async def delete_model(request: DeleteRequest, http_request: Request) -> dict:
    """Delete a model."""
    model_manager._load_metadata()
    if not model_manager.is_model_installed(request.name):
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.name}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    if model_manager.get_model_type(request.name) == "decision":
        from mlx_serve.core.runtime_topology import retrieval_worker_isolation_enabled
        from mlx_serve.routers.decision import delete_decision_model

        worker = await _decision_worker_request(
            http_request, "/internal/decision/delete", {"name": request.name}
        )
        if worker is None and retrieval_worker_isolation_enabled():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "Decision worker is unavailable",
                        "code": "worker_unavailable",
                    }
                },
            )
        try:
            deleted = (
                worker["deleted"]
                if worker is not None
                else await delete_decision_model(request.name)
            )
        except ModelInUseError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": str(exc), "code": "model_in_use"}},
            ) from exc
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Model '{request.name}' not found",
                        "code": "model_not_found",
                    }
                },
            )
        if worker is not None:
            model_manager._load_metadata()
    else:
        model_manager.delete_model(request.name)
    return {"status": "success"}


@router.get("/v1/models/cache")
async def get_model_cache_stats(http_request: Request) -> dict:
    """Return loaded model cache and memory diagnostics."""
    worker = await _decision_worker_request(http_request, "/internal/decision/cache")
    return _merge_decision_cache(model_manager.get_cache_stats(), worker)


@router.post("/v1/models/unload", response_model=UnloadResponse)
async def unload_models(request: UnloadRequest, http_request: Request) -> UnloadResponse:
    """Unload cached idle models without deleting them from disk."""
    if not request.all:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Set 'all' to true to unload all cached models",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )

    result = model_manager.unload_all(request.type)
    worker = None
    if request.type in (None, "decision"):
        worker = await _decision_worker_request(http_request, "/internal/decision/unload", {})
        if worker is not None:
            result["unloaded"].extend(worker["unloaded"])
            result["skipped_active"].extend(worker["skipped_active"])
    return UnloadResponse(
        unloaded=result["unloaded"],
        skipped_active=result["skipped_active"],
        cache=_merge_decision_cache(model_manager.get_cache_stats(), worker and worker["cache"]),
    )


@router.post("/v1/models/{model_name}/unload", response_model=UnloadResponse)
async def unload_model(
    model_name: str,
    http_request: Request,
    model_type: ModelType | None = None,
    type_: ModelType | None = Query(default=None, alias="type"),
) -> UnloadResponse:
    """Unload one cached model without deleting it from disk."""
    target_type = type_ or model_type
    _, _, alias_type = resolve_model_alias(model_name)
    is_decision = target_type == "decision" or (
        target_type is None
        and (alias_type == "decision" or model_manager.get_model_type(model_name) == "decision")
    )
    if is_decision:
        worker = await _decision_worker_request(
            http_request, "/internal/decision/unload", {"name": model_name}
        )
        if worker is not None:
            return UnloadResponse(
                unloaded=worker["unloaded"],
                skipped_active=worker["skipped_active"],
                cache=_merge_decision_cache(model_manager.get_cache_stats(), worker["cache"]),
            )
    try:
        unloaded = model_manager.unload_model(model_name, target_type)
    except ModelInUseError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "model_in_use",
                }
            },
        ) from e

    return UnloadResponse(
        unloaded=unloaded,
        cache=model_manager.get_cache_stats(),
    )


@router.post("/api/show", response_model=ShowResponse)
async def show_model(request: ShowRequest) -> ShowResponse:
    """Get detailed information about a model."""
    info = model_manager.get_model_info(request.name)

    if info is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.name}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    return ShowResponse(
        name=info.name,
        type=info.model_type,
        size=info.size,
        modified_at=info.modified_at,
        hf_repo=info.hf_repo,
        path=str(info.path),
    )
