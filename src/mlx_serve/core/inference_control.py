"""Admission control for model inference."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from mlx_serve.config import settings
from mlx_serve.core.model_manager import resolve_model_alias


class InferenceOverloadedError(RuntimeError):
    """Raised when a request cannot be admitted for inference."""


@dataclass
class _InferenceState:
    """Mutable admission-control state for a single model key."""

    active: int = 0
    queue: list[Any] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class InferenceLease:
    """A held inference slot that must be released."""

    def __init__(
        self,
        controller: "InferenceAdmissionController",
        key: str,
        state: _InferenceState,
    ):
        self._controller = controller
        self._key = key
        self._state = state
        self._released = False

    async def release(self) -> None:
        """Release the held inference slot."""
        if self._released:
            return
        self._released = True
        await self._controller._release(self._state)

    async def __aenter__(self) -> "InferenceLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class InferenceAdmissionController:
    """FIFO per-model admission control."""

    def __init__(
        self,
        max_concurrency: int | None = None,
        max_queue_size: int | None = None,
        acquire_timeout_seconds: float | None = None,
    ):
        self.max_concurrency = (
            settings.inference_max_concurrency_per_model
            if max_concurrency is None
            else max_concurrency
        )
        self.max_queue_size = (
            settings.inference_max_queue_per_model
            if max_queue_size is None
            else max_queue_size
        )
        self.acquire_timeout_seconds = (
            settings.inference_queue_timeout_seconds
            if acquire_timeout_seconds is None
            else acquire_timeout_seconds
        )
        self._states: dict[str, _InferenceState] = {}
        self._states_lock = threading.Lock()

    def _get_state(self, key: str) -> _InferenceState:
        with self._states_lock:
            state = self._states.get(key)
            if state is None:
                state = _InferenceState()
                self._states[key] = state
            return state

    async def acquire(self, key: str) -> InferenceLease:
        """Acquire a model slot or raise when overloaded."""
        state = self._get_state(key)
        token = object()

        async with state.condition:
            should_queue = state.active >= self.max_concurrency or bool(state.queue)
            if should_queue:
                if self.max_queue_size is not None and len(state.queue) >= self.max_queue_size:
                    raise InferenceOverloadedError(
                        f"Model '{key}' is overloaded: queue is full"
                    )

                state.queue.append(token)
                try:
                    waiter = state.condition.wait_for(
                        lambda: state.queue and state.queue[0] is token and state.active < self.max_concurrency
                    )
                    if self.acquire_timeout_seconds is None:
                        await waiter
                    else:
                        await asyncio.wait_for(waiter, timeout=self.acquire_timeout_seconds)
                except asyncio.TimeoutError as exc:
                    if token in state.queue:
                        state.queue.remove(token)
                        state.condition.notify_all()
                    raise InferenceOverloadedError(
                        f"Model '{key}' is overloaded: wait timed out after "
                        f"{self.acquire_timeout_seconds:.1f}s"
                    ) from exc
                except BaseException:
                    if token in state.queue:
                        state.queue.remove(token)
                        state.condition.notify_all()
                    raise

                if state.queue and state.queue[0] is token:
                    state.queue.pop(0)

            state.active += 1

        return InferenceLease(self, key, state)

    async def _release(self, state: _InferenceState) -> None:
        async with state.condition:
            if state.active > 0:
                state.active -= 1
            state.condition.notify_all()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a best-effort snapshot for debugging and tests."""
        with self._states_lock:
            items = list(self._states.items())
        return {
            key: {"active": state.active, "queued": len(state.queue)}
            for key, state in items
        }

    def reset(self) -> None:
        """Clear tracked states. Intended for tests."""
        with self._states_lock:
            self._states.clear()


def build_inference_key(model_type: str, model_name: str) -> str:
    """Build a stable per-model inference key."""
    resolved_name, _, resolved_type = resolve_model_alias(model_name)
    return f"{resolved_type or model_type}:{resolved_name}"


def build_overload_detail(message: str) -> dict:
    """Build an OpenAI-style overload error payload."""
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "code": "server_overloaded",
        }
    }


inference_controller = InferenceAdmissionController()
