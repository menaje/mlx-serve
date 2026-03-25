"""Tests for request batching."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_serve.core.batch_processor import EmbeddingBatchProcessor


@pytest.mark.asyncio
async def test_embedding_batch_processor_batches_whole_requests(monkeypatch):
    """Concurrent embedding requests should be combined into a single model call."""
    calls: list[list[str]] = []

    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.array = lambda value: value
    fake_mlx.core = fake_mlx_core

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append(list(texts))
            return {
                "input_ids": [[index] for index, _ in enumerate(texts)],
                "attention_mask": [[1] for _ in texts],
            }

    class FakeModel:
        def __call__(self, input_ids, attention_mask=None):
            return SimpleNamespace(
                text_embeds=np.array([[float(row[0])] for row in input_ids])
            )

    processor = EmbeddingBatchProcessor(model=FakeModel(), tokenizer=FakeTokenizer())
    try:
        task1 = asyncio.create_task(processor.embed(["a", "b"]))
        task2 = asyncio.create_task(processor.embed(["c"]))

        result1, result2 = await asyncio.gather(task1, task2)

        assert calls == [["a", "b", "c"]]
        assert result1 == [[0.0], [1.0]]
        assert result2 == [[2.0]]
    finally:
        await processor.stop()
