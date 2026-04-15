"""Admission control for model inference."""

import asyncio
import threading
from contextlib import asynccontextmanager
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
        raise_if_server_overloaded()
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
                    def can_activate() -> bool:
                        return (
                            bool(state.queue)
                            and state.queue[0] is token
                            and state.active < self.max_concurrency
                        )

                    waiter = state.condition.wait_for(
                        can_activate
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

            raise_if_server_overloaded()
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
        execution_locks.reset()


class ModelExecutionLocks:
    """Async locks for model-bound operations that must not overlap."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> asyncio.Lock:
        """Get or create a per-model execution lock."""
        with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def hold(self, key: str):
        """Hold a per-model execution lock."""
        async with self.get(key):
            yield

    def reset(self) -> None:
        """Clear tracked locks. Intended for tests."""
        with self._lock:
            self._locks.clear()


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


def get_model_execution_lock(key: str) -> asyncio.Lock:
    """Return the shared execution lock for a model key."""
    return execution_locks.get(key)


def raise_if_server_overloaded() -> None:
    """Reject new work when the system-wide admission guard is active."""
    from mlx_serve.core.system_guard import memory_monitor

    reason = memory_monitor.overload_reason()
    if reason:
        raise InferenceOverloadedError(reason)


inference_controller = InferenceAdmissionController()
execution_locks = ModelExecutionLocks()
