"""System One decision API for native MLX decision models."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlx_serve.config import settings
from mlx_serve.core.d1_decision import DecisionInputError
from mlx_serve.core.default_models import clear_deleted_decision_default, request_model_default
from mlx_serve.core.inference_control import (
    InferenceOverloadedError,
    build_inference_key,
    build_overload_detail,
    get_model_execution_lock,
    inference_controller,
)
from mlx_serve.core.mlx_memory import clear_mlx_cache
from mlx_serve.core.model_manager import ModelInUseError, model_manager, resolve_model_alias
from mlx_serve.core.model_memory import ModelLoadMemoryError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["decision"])
internal_router = APIRouter()


class InternalDecisionUnload(BaseModel):
    name: str | None = None


class InternalDecisionDelete(BaseModel):
    name: str


async def delete_decision_model(name: str) -> bool:
    """Delete a decision model in the process that owns its inference cache."""
    key = build_inference_key("decision", name)
    lock = get_model_execution_lock(key)
    if lock.locked() or key in inference_controller.active_keys():
        raise ModelInUseError(f"Model '{name}' is active and cannot be deleted")

    def delete_from_disk() -> bool:
        # Share the publication lock with decision model installs, and reload the
        # registry so deleting here cannot overwrite a newer gateway registration.
        lock_path = settings.models_dir.parent / ".decision-metadata.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as registry_lock:
            fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
            try:
                model_manager._load_metadata()
                if not model_manager.is_model_installed(name):
                    return False
                if model_manager.get_model_type(name) != "decision":
                    return False
                deleted = model_manager.delete_model(name)
                if deleted:
                    # A damaged defaults file must not turn a successful delete
                    # into a failure; stale selections are checked on every use.
                    with contextlib.suppress(OSError, ValueError):
                        clear_deleted_decision_default(name, settings.models_dir)
                return deleted
            finally:
                fcntl.flock(registry_lock.fileno(), fcntl.LOCK_UN)

    async with lock:
        operation = asyncio.create_task(asyncio.to_thread(delete_from_disk))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Keep the execution lock until the deletion thread really stops.
            try:
                await operation
            except Exception:
                logger.exception("Decision model deletion failed after cancellation")
            raise


@internal_router.get("/internal/decision/cache")
async def decision_worker_cache() -> dict[str, Any]:
    """Expose the decision worker's cache to its local gateway."""
    return model_manager.get_cache_stats()


@internal_router.post("/internal/decision/unload")
async def decision_worker_unload(request: InternalDecisionUnload) -> dict[str, Any]:
    """Unload idle models within the worker that actually owns them."""
    try:
        if request.name is None:
            result = model_manager.unload_all("decision")
        else:
            result = {
                "unloaded": model_manager.unload_model(request.name, "decision"),
                "skipped_active": [],
            }
    except ModelInUseError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"message": str(exc), "code": "model_in_use"}},
        ) from exc
    return {**result, "cache": model_manager.get_cache_stats()}


@internal_router.post("/internal/decision/delete")
async def decision_worker_delete(request: InternalDecisionDelete) -> dict[str, bool]:
    """Remove an idle model and its cache from the isolated worker."""
    try:
        deleted = await delete_decision_model(request.name)
    except ModelInUseError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"message": str(exc), "code": "model_in_use"}},
        ) from exc
    return {"deleted": deleted}


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "bool", "noul", "score"]
    instructions: Any = None
    criteria: Any = None

    @model_validator(mode="after")
    def validate_criteria(self) -> "DecisionQuestion":
        criteria = self.criteria
        if self.type == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("choice criteria must contain 2 to 255 named options")
            if any(not isinstance(key, str) or not key for key in criteria):
                raise ValueError("choice option names must be nonempty strings")
        elif self.type in ("noul", "bool"):
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
            ):
                raise ValueError("noul criteria may contain only false and true descriptions")
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must contain 2 to 10 ordered levels")
            levels = [
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in criteria
            ]
            if any(not level for level in levels) or len(set(levels)) != len(levels):
                raise ValueError("score levels must be nonempty and distinct")
        try:
            json.dumps({"instructions": self.instructions, "criteria": criteria}, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("question values must be finite JSON") from exc
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    state: Any
    questions: dict[str, DecisionQuestion] = Field(min_length=1, max_length=64)
    images: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def use_default_model(cls, data: Any) -> Any:
        return request_model_default(data, "decision")

    @field_validator("questions")
    @classmethod
    def validate_question_names(cls, questions: dict[str, DecisionQuestion]):
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name) for name in questions):
            raise ValueError(
                "question names must contain 1 to 100 ASCII letters, digits, _, . or -"
            )
        return questions

    @model_validator(mode="after")
    def validate_state_and_media(self) -> "DecisionRequest":
        if self.state is None or self.state == "":
            raise ValueError("state must not be empty")
        if self.images or self.videos:
            raise ValueError("image and video decision inputs are not supported yet")
        try:
            json.dumps(self.state, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must be finite JSON") from exc
        return self


def _run_decision(request: DecisionRequest, *, typed: bool = False) -> dict[str, Any]:
    model = model_manager.get_decision_model(request.model)
    try:
        questions = {name: question.model_dump() for name, question in request.questions.items()}
        if typed and hasattr(model, "decide"):
            return model.decide(request.state, questions)
        for question in questions.values():
            if question["type"] == "bool":
                question["type"] = "noul"
        result = model.system_one(request.state, questions)
        if not typed:
            return result
        answers = {}
        for name, answer in result["answers"].items():
            if "error" in answer:
                raise DecisionInputError(answer.get("message", answer["error"]))
            if answer["type"] == "noul":
                probability = answer["noul"]
                answers[name] = {
                    "type": "bool",
                    "value": probability >= 0.5,
                    "probability": probability,
                }
            else:
                probabilities = answer["probabilities"]
                metadata = {"confidence": max(probabilities.values())}
                if answer["type"] == "score":
                    metadata["legend"] = answer["legend"]
                answers[name] = {
                    "type": answer["type"],
                    "value": answer["choice"] if answer["type"] == "choice" else answer["score"],
                    "probabilities": probabilities,
                    "metadata": metadata,
                }
        return {"answers": answers, "usage": result["usage"]}
    finally:
        if settings.retrieval_clear_mlx_cache_after_request:
            clear_mlx_cache(log=logger, reason="decision request")


async def _handle_decision(request: DecisionRequest, *, typed: bool) -> dict[str, Any]:
    key = build_inference_key("decision", request.model)
    try:
        lease = await inference_controller.acquire(key)
    except InferenceOverloadedError as exc:
        raise HTTPException(status_code=503, detail=build_overload_detail(str(exc))) from exc

    async def run_with_lease() -> dict[str, Any]:
        async with lease:
            async with get_model_execution_lock(key):
                task = asyncio.create_task(asyncio.to_thread(_run_decision, request, typed=typed))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    try:
                        await task
                    except Exception as exc:
                        logger.error(
                            "Decision inference failed after the request was cancelled (%s)",
                            type(exc).__name__,
                        )
                    raise

    running = asyncio.create_task(run_with_lease())

    def observe_background(task: asyncio.Task) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error(
                "Decision inference failed after the request ended (%s)", type(exc).__name__
            )

    try:
        result = await asyncio.wait_for(
            asyncio.shield(running), timeout=settings.decision_execution_timeout_seconds
        )
    except TimeoutError as exc:
        # The request ends promptly, but the lease remains held until the thread actually stops.
        running.add_done_callback(observe_background)
        raise HTTPException(
            status_code=504,
            detail={
                "error": {"message": "Decision inference timed out", "code": "decision_timeout"}
            },
        ) from exc
    except asyncio.CancelledError:
        running.add_done_callback(observe_background)
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail={"error": {"message": str(exc), "code": "model_not_found"}}
        ) from exc
    except ModelLoadMemoryError as exc:
        raise HTTPException(status_code=503, detail=build_overload_detail(str(exc))) from exc
    except DecisionInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": str(exc), "code": "decision_input_too_long"}},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc), "code": "invalid_decision_model"}},
        ) from exc
    if typed:
        canonical_name, _, _ = resolve_model_alias(request.model)
        revision = model_manager._metadata.get(canonical_name, {}).get("revision")
        return {"model": canonical_name, "revision": revision, **result}
    return {"model": request.model, **result}


@router.post("/v1/systemone")
async def system_one(request: DecisionRequest) -> dict[str, Any]:
    """Answer decisions using the Ollama System One response format."""
    return await _handle_decision(request, typed=False)


@router.post("/v1/decisions")
async def decisions(request: DecisionRequest) -> dict[str, Any]:
    """Answer typed decisions using the model's official prediction contract."""
    return await _handle_decision(request, typed=True)
