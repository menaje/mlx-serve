"""Tests for request batching."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_serve.core.batch_processor import EmbeddingBatchProcessor, RerankBatchProcessor


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


@pytest.mark.asyncio
async def test_embedding_batch_processor_limits_total_texts_per_batch(monkeypatch):
    """Concurrent embedding requests should respect the per-batch text budget."""
    from mlx_serve.config import settings

    calls: list[list[str]] = []
    monkeypatch.setattr(settings, "embedding_batch_max_texts", 2)

    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.array = lambda value: value
    fake_mlx.core = fake_mlx_core

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    token_map = {"a": 10, "b": 20, "c": 30}

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append(list(texts))
            return {
                "input_ids": [[token_map[text]] for text in texts],
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

        assert calls == [["a", "b"], ["c"]]
        assert result1 == [[10.0], [20.0]]
        assert result2 == [[30.0]]
    finally:
        await processor.stop()


@pytest.mark.asyncio
async def test_embedding_batch_processor_chunks_large_single_request(monkeypatch):
    """A single large embedding request should be split into bounded chunks."""
    from mlx_serve.config import settings

    calls: list[list[str]] = []
    monkeypatch.setattr(settings, "embedding_batch_max_texts", 2)

    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.array = lambda value: value
    fake_mlx.core = fake_mlx_core

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    token_map = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append(list(texts))
            return {
                "input_ids": [[token_map[text]] for text in texts],
                "attention_mask": [[1] for _ in texts],
            }

    class FakeModel:
        def __call__(self, input_ids, attention_mask=None):
            return SimpleNamespace(
                text_embeds=np.array([[float(row[0])] for row in input_ids])
            )

    processor = EmbeddingBatchProcessor(model=FakeModel(), tokenizer=FakeTokenizer())
    try:
        result = await processor.embed(["a", "b", "c", "d", "e"])

        assert calls == [["a", "b"], ["c", "d"], ["e"]]
        assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    finally:
        await processor.stop()


def test_rerank_batch_processor_respects_document_limit(monkeypatch):
    """Rerank micro-batches should split once the document cap is reached."""
    from mlx_serve.config import settings

    monkeypatch.setattr(settings, "rerank_batch_max_documents", 2)
    monkeypatch.setattr(settings, "rerank_batch_max_tokens", 100)

    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.array = lambda value: np.array(value)
    fake_mlx_core.stack = np.stack
    fake_mlx_core.softmax = lambda value: np.exp(value) / np.exp(value).sum()
    fake_mlx.core = fake_mlx_core

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    calls: list[tuple[int, int]] = []

    class FakeTokenizer:
        pad_token_id = 0
        bos_token_id = None
        eos_token_id = None

        def encode(self, prompt, add_special_tokens=False):
            length = int(prompt.rsplit("len=", 1)[1])
            return list(range(1, length + 1))

        def convert_tokens_to_ids(self, token):
            return {"no": 0, "yes": 1}[token]

    class FakeModel:
        def __call__(self, input_ids):
            batch, seq_len = input_ids.shape
            calls.append((batch, seq_len))
            logits = np.zeros((batch, seq_len, 2), dtype=float)
            for row_index, row in enumerate(input_ids):
                prompt_len = int(np.count_nonzero(row))
                logits[row_index, prompt_len - 1, 1] = float(prompt_len)
            return SimpleNamespace(logits=logits)

    processor = RerankBatchProcessor(model=FakeModel(), tokenizer=FakeTokenizer())
    scores, total_tokens = processor.compute_scores(
        query="q",
        documents=["len=2", "len=3", "len=4"],
    )

    assert calls == [(2, 3), (1, 4)]
    assert len(scores) == 3
    assert total_tokens == 9
    assert scores[0] < scores[1] < scores[2]


def test_rerank_batch_processor_respects_token_budget(monkeypatch):
    """Rerank micro-batches should split when padded token budget would overflow."""
    from mlx_serve.config import settings

    monkeypatch.setattr(settings, "rerank_batch_max_documents", 4)
    monkeypatch.setattr(settings, "rerank_batch_max_tokens", 5)

    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.array = lambda value: np.array(value)
    fake_mlx_core.stack = np.stack
    fake_mlx_core.softmax = lambda value: np.exp(value) / np.exp(value).sum()
    fake_mlx.core = fake_mlx_core

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    calls: list[tuple[int, int]] = []

    class FakeTokenizer:
        pad_token_id = 0
        bos_token_id = None
        eos_token_id = None

        def encode(self, prompt, add_special_tokens=False):
            length = int(prompt.rsplit("len=", 1)[1])
            return list(range(1, length + 1))

        def convert_tokens_to_ids(self, token):
            return {"no": 0, "yes": 1}[token]

    class FakeModel:
        def __call__(self, input_ids):
            batch, seq_len = input_ids.shape
            calls.append((batch, seq_len))
            logits = np.zeros((batch, seq_len, 2), dtype=float)
            for row_index, row in enumerate(input_ids):
                prompt_len = int(np.count_nonzero(row))
                logits[row_index, prompt_len - 1, 1] = float(prompt_len)
            return SimpleNamespace(logits=logits)

    processor = RerankBatchProcessor(model=FakeModel(), tokenizer=FakeTokenizer())
    scores, total_tokens = processor.compute_scores(
        query="q",
        documents=["len=3", "len=3"],
    )

    assert calls == [(1, 3), (1, 3)]
    assert len(scores) == 2
    assert total_tokens == 6
