"""Models API router - OpenAI and Ollama compatible."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import ModelType, model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


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


class DeleteRequest(BaseModel):
    """Model delete request."""

    name: str = Field(..., description="Model name to delete")


class ShowRequest(BaseModel):
    """Model show request."""

    name: str = Field(..., description="Model name to show")


class ShowResponse(BaseModel):
    """Model show response."""

    name: str
    type: ModelType
    size: int
    modified_at: str
    hf_repo: str
    path: str


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
    """Download and convert a model from Hugging Face."""
    import json

    async def generate():
        async for status in model_manager.pull_model(
            hf_repo=request.name,
            model_type=request.type,
        ):
            yield json.dumps(status) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@router.delete("/api/delete")
async def delete_model(request: DeleteRequest) -> dict:
    """Delete a model."""
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

    model_manager.delete_model(request.name)
    return {"status": "success"}


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
