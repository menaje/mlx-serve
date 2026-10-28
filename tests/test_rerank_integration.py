"""Optional real Qwen3-Reranker regression using an existing local MLX checkpoint."""

import os

import pytest

from mlx_serve.core.batch_processor import RerankBatchProcessor
from mlx_serve.routers.rerank import compute_rerank_score


@pytest.mark.skipif(
    not os.environ.get("MLX_SERVE_RERANK_TEST_MODEL_DIR"),
    reason="Set MLX_SERVE_RERANK_TEST_MODEL_DIR to a local Qwen3-Reranker checkpoint",
)
def test_qwen3_reranks_relevant_document_and_single_score_matches_batch():
    from mlx_lm import load

    model, tokenizer = load(os.environ["MLX_SERVE_RERANK_TEST_MODEL_DIR"])
    query = "What is the capital of China?"
    relevant = "The capital of China is Beijing."
    unrelated = "A cat sleeps on the sofa."
    batch, tokens = RerankBatchProcessor(model, tokenizer).compute_scores(
        query, [relevant, unrelated]
    )
    single = compute_rerank_score(model, tokenizer, query, relevant)
    assert 0 <= batch[1] < batch[0] <= 1
    assert batch[0] - batch[1] > 0.25
    assert single == pytest.approx(batch[0], abs=0.002)
    assert tokens > 0
