"""Tests for inference admission control."""

import asyncio

import pytest

from mlx_serve.core.inference_control import (
    InferenceAdmissionController,
    InferenceOverloadedError,
    build_inference_key,
)


@pytest.mark.asyncio
async def test_inference_controller_queues_then_releases():
    """A second request should wait in queue until the active one completes."""
    controller = InferenceAdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        acquire_timeout_seconds=1.0,
    )

    lease1 = await controller.acquire("llm:test-model")
    waiter = asyncio.create_task(controller.acquire("llm:test-model"))

    await asyncio.sleep(0)
    assert controller.snapshot()["llm:test-model"] == {"active": 1, "queued": 1}

    await lease1.release()
    lease2 = await asyncio.wait_for(waiter, timeout=1.0)
    assert controller.snapshot()["llm:test-model"] == {"active": 1, "queued": 0}

    await lease2.release()
    assert controller.snapshot()["llm:test-model"] == {"active": 0, "queued": 0}


@pytest.mark.asyncio
async def test_inference_controller_rejects_when_queue_full():
    """Requests beyond the configured queue depth should be rejected."""
    controller = InferenceAdmissionController(
        max_concurrency=1,
        max_queue_size=0,
        acquire_timeout_seconds=1.0,
    )

    lease = await controller.acquire("llm:test-model")
    with pytest.raises(InferenceOverloadedError, match="queue is full"):
        await controller.acquire("llm:test-model")

    await lease.release()


@pytest.mark.asyncio
async def test_inference_controller_times_out_waiters():
    """Queued requests should fail when they wait too long for a slot."""
    controller = InferenceAdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        acquire_timeout_seconds=0.01,
    )

    lease = await controller.acquire("llm:test-model")
    with pytest.raises(InferenceOverloadedError, match="wait timed out"):
        await controller.acquire("llm:test-model")

    await lease.release()


@pytest.mark.asyncio
async def test_inference_controller_preserves_fifo_order():
    """Queued requests should acquire slots in arrival order."""
    controller = InferenceAdmissionController(
        max_concurrency=1,
        max_queue_size=None,
        acquire_timeout_seconds=None,
    )

    lease1 = await controller.acquire("llm:test-model")
    waiter2 = asyncio.create_task(controller.acquire("llm:test-model"))
    waiter3 = asyncio.create_task(controller.acquire("llm:test-model"))

    await asyncio.sleep(0)
    assert controller.snapshot()["llm:test-model"] == {"active": 1, "queued": 2}

    await lease1.release()
    lease2 = await asyncio.wait_for(waiter2, timeout=1.0)
    assert not waiter3.done()

    await lease2.release()
    lease3 = await asyncio.wait_for(waiter3, timeout=1.0)
    await lease3.release()

    assert controller.snapshot()["llm:test-model"] == {"active": 0, "queued": 0}


def test_build_inference_key_resolves_aliases():
    """Alias resolution should collapse to the canonical model key."""
    assert build_inference_key("llm", "qwen-embedding") == "embedding:Qwen3-Embedding-0.6B"
    assert build_inference_key("llm", "custom-model") == "llm:custom-model"
