"""Local operational state and commands for the native application."""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mlx_serve.config import settings
from mlx_serve.core.default_models import (
    DEFAULT_TYPES,
    DefaultSelectionChanged,
    default_model,
    set_default_model,
)
from mlx_serve.core.inference_control import (
    build_inference_key,
    get_model_execution_lock,
    inference_controller,
)
from mlx_serve.core.local_access import internal_management_headers
from mlx_serve.core.managed_config import settings_snapshot
from mlx_serve.core.managed_originals import (
    cleanup_unreferenced_originals,
    list_unreferenced_originals,
)
from mlx_serve.core.mlx_memory import get_mlx_memory_snapshot
from mlx_serve.core.model_manager import ModelInUseError, ModelType, model_manager
from mlx_serve.core.runtime_topology import (
    get_generation_worker_kind,
    get_retrieval_worker_kind,
)
from mlx_serve.core.storage_access import model_storage_available
from mlx_serve.core.system_guard import memory_monitor
from mlx_serve.routers.retrieval_proxy import perform_worker_request

router = APIRouter(tags=["local management"])
internal_router = APIRouter()


class ModelOperation(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    type: ModelType


class OriginalCleanupEntry(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*-[0-9a-f]{40}$", max_length=300)
    size_bytes: int = Field(ge=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class OriginalCleanupRequest(BaseModel):
    entries: list[OriginalCleanupEntry] = Field(min_length=1, max_length=100)


def _local_snapshot() -> dict[str, Any]:
    return {
        "cache": model_manager.get_cache_stats(),
        "requests": inference_controller.snapshot(),
        "memory": memory_monitor.health_payload(),
        "mlx_memory": get_mlx_memory_snapshot(),
    }


@internal_router.get("/internal/management/state")
async def internal_state() -> dict[str, Any]:
    return _local_snapshot()


async def _local_operation(operation: str, payload: ModelOperation) -> dict[str, Any]:
    worker_kind = get_retrieval_worker_kind() or get_generation_worker_kind()
    if worker_kind and payload.type != worker_kind:
        raise HTTPException(status_code=400, detail="Model type does not belong to this worker")
    model_manager._load_metadata()
    if not model_manager.is_model_installed(payload.name):
        raise HTTPException(status_code=404, detail="Model is not installed")
    if model_manager.get_model_type(payload.name) != payload.type:
        raise HTTPException(status_code=400, detail="Model type does not match the registry")
    key = build_inference_key(payload.type, payload.name)
    lock = get_model_execution_lock(key)
    if lock.locked() or key in inference_controller.active_keys():
        raise HTTPException(status_code=409, detail="Model is processing a request")
    async with lock:
        if operation == "load":
            loaded = await asyncio.to_thread(model_manager.preload_model, payload.name)
            if not loaded:
                raise HTTPException(status_code=422, detail="Model could not be loaded; check logs")
            return {"loaded": True, "state": _local_snapshot()}
        try:
            unloaded = model_manager.unload_model(payload.name, payload.type)
        except ModelInUseError as exc:
            raise HTTPException(status_code=409, detail="Model is processing a request") from exc
        return {"unloaded": unloaded, "state": _local_snapshot()}


@internal_router.post("/internal/management/load")
async def internal_load(payload: ModelOperation) -> dict[str, Any]:
    return await _local_operation("load", payload)


@internal_router.post("/internal/management/unload")
async def internal_unload(payload: ModelOperation) -> dict[str, Any]:
    return await _local_operation("unload", payload)


async def _worker_request(url: str, path: str, payload: dict | None = None) -> dict[str, Any]:
    try:
        response = await asyncio.to_thread(
            perform_worker_request, url + path, "GET" if payload is None else "POST",
            b"" if payload is None else json.dumps(payload).encode(),
            {"Content-Type": "application/json", **internal_management_headers()},
            timeout=3 if payload is None else 180, loopback_only=True,
        )
        data = json.loads(response.body)
    except (OSError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Worker is unavailable") from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=data.get("detail", data))
    return data


@router.get("/management/settings")
async def applied_settings() -> dict[str, Any]:
    return settings_snapshot(applied=settings)


@router.get("/management/originals")
async def original_cleanup_candidates() -> dict[str, Any]:
    try:
        candidates = await asyncio.to_thread(list_unreferenced_originals, settings.models_dir)
        return {"candidates": candidates}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/management/originals/cleanup")
async def cleanup_originals(payload: OriginalCleanupRequest) -> dict[str, int]:
    try:
        count = await asyncio.to_thread(
            cleanup_unreferenced_originals, settings.models_dir,
            [entry.model_dump() for entry in payload.entries],
        )
        return {"removed": count}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/management/state")
async def management_state(request: Request) -> dict[str, Any]:
    """Keep each worker's cache and memory separate and report unreachable workers."""
    local = _local_snapshot()
    workers: dict[str, Any] = {}
    if settings.retrieval_worker_isolation_enabled:
        urls = getattr(request.app.state, "retrieval_worker_urls", {})
        for kind in ("embedding", "reranker", "decision"):
            try:
                if kind not in urls:
                    raise HTTPException(status_code=503)
                workers[kind] = {"available": True, **await _worker_request(
                    urls[kind], "/internal/management/state"
                )}
            except HTTPException:
                workers[kind] = {"available": False}
    generation = getattr(request.app.state, "generation_worker_supervisor", None)
    if settings.generation_worker_isolation_enabled:
        snapshots = generation.snapshot() if generation is not None else {}
        for key, worker in snapshots.items():
            try:
                if not worker.get("alive"):
                    raise HTTPException(status_code=503)
                workers[f"generation:{key}"] = {"available": True, **await _worker_request(
                    str(worker["url"]), "/internal/management/state"
                )}
            except (HTTPException, KeyError):
                workers[f"generation:{key}"] = {"available": False}
    storage_available = model_storage_available(settings.models_dir)
    defaults = {kind: default_model(kind) for kind in DEFAULT_TYPES} if storage_available else {}
    if storage_available:
        model_manager._load_metadata()
    models = [] if storage_available else None
    for model in model_manager.list_models() if storage_available else []:
        metadata = model_manager._metadata.get(model.name, {})
        owner = (
            workers.get(model.model_type) if settings.retrieval_worker_isolation_enabled else None
        )
        if settings.generation_worker_isolation_enabled and model.model_type in ("llm", "vlm"):
            # A generation worker can be model-scoped; absence is not evidence of a load.
            candidates = [value for key, value in workers.items() if key.startswith("generation:")]
            known = [value for value in candidates if value.get("available")]
            loaded = any(
                model.name in value["cache"][f"{model.model_type}_models"]["models"]
                for value in known
            )
            load_state = (
                "loaded" if loaded else
                ("not_loaded" if len(known) == len(candidates) else "unknown")
            )
            request_state = "unknown"
        else:
            source = owner if owner is not None else local
            if owner is not None and not owner.get("available"):
                load_state, request_state = "unknown", "unknown"
            else:
                cache = source["cache"].get(f"{model.model_type}_models", {})
                load_state = "loaded" if model.name in cache.get("models", []) else "not_loaded"
                key = build_inference_key(model.model_type, model.name)
                active = source["requests"].get(key, {})
                request_state = "running" if active.get("active") else (
                    "queued" if active.get("queued") else "idle"
                )
        models.append({
            "name": model.name, "type": model.model_type, "source": model.hf_repo,
            "size_bytes": model.size, "path": str(model.path), "metadata": metadata,
            "file_state": "installed", "load_state": load_state, "request_state": request_state,
            "is_default": defaults.get(model.model_type) == model.name,
        })
    disk = None
    if storage_available:
        try:
            usage = shutil.disk_usage(settings.models_dir)
            disk = {"total_bytes": usage.total, "free_bytes": usage.free}
        except OSError:
            pass
    return {
        "state": "degraded" if not storage_available or any(
            not worker["available"] for worker in workers.values()
        )
        else "healthy",
        "models": models, "gateway": local, "workers": workers, "disk": disk,
        "storage": {"state": "available" if storage_available else "unavailable"},
        "default_models_supported": True,
    }


@router.post("/management/models/{operation}")
async def manage_model(operation: str, payload: ModelOperation, request: Request) -> dict[str, Any]:
    if operation not in ("load", "unload", "delete", "set-default", "clear-default"):
        raise HTTPException(status_code=404, detail="Unknown model operation")
    if not model_storage_available(settings.models_dir):
        raise HTTPException(status_code=503, detail="Model storage is unavailable")
    if operation in ("set-default", "clear-default"):
        if payload.type not in DEFAULT_TYPES:
            raise HTTPException(
                status_code=422, detail="Default selection is not supported for this model type"
            )
        try:
            set_default_model(
                payload.type, payload.name if operation == "set-default" else None,
                expected_name=payload.name if operation == "clear-default" else None,
            )
        except DefaultSelectionChanged as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"default": payload.name if operation == "set-default" else None}
    if operation == "delete":
        # The decision route serializes deletion with the owning worker and
        # preparation registry. Other model types do not have that guarantee yet.
        model_manager._load_metadata()
        if payload.type != "decision" or model_manager.get_model_type(payload.name) != "decision":
            raise HTTPException(status_code=501, detail="Safe deletion is supported for d1 only")
        from mlx_serve.routers.models import DeleteRequest, delete_model

        return await delete_model(DeleteRequest(name=payload.name), request)
    if (
        payload.type in ("embedding", "reranker", "decision")
        and settings.retrieval_worker_isolation_enabled
    ):
        url = getattr(request.app.state, "retrieval_worker_urls", {}).get(payload.type)
        if not url:
            raise HTTPException(status_code=503, detail="Worker is unavailable")
        return await _worker_request(url, f"/internal/management/{operation}", payload.model_dump())
    if payload.type in ("llm", "vlm") and settings.generation_worker_isolation_enabled:
        raise HTTPException(status_code=501, detail="Generation worker management is not available")
    return await _local_operation(operation, payload)
