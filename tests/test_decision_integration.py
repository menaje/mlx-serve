"""Optional real-model comparison against an Ollama D3 reference result."""

import os
from pathlib import Path

import pytest

from mlx_serve.core.decision import D3DecisionModel


@pytest.mark.skipif(
    not os.environ.get("MLX_SERVE_D3_TEST_MODEL_DIR"),
    reason="Set MLX_SERVE_D3_TEST_MODEL_DIR to a local d3-lite checkpoint",
)
def test_native_mlx_matches_ollama_reference():
    model = D3DecisionModel(Path(os.environ["MLX_SERVE_D3_TEST_MODEL_DIR"]))
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments and refunds", "technical": "Software errors"},
        },
        "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
    }
    result = model.system_one("The customer was charged twice and asks for a refund.", questions)
    assert result["answers"]["route"]["choice"] == "billing"
    assert result["answers"]["route"]["probabilities"]["billing"] == pytest.approx(
        0.99714449, abs=0.01
    )
    assert result["answers"]["refund"]["noul"] == pytest.approx(0.98668019, abs=0.01)
    assert result["usage"]["input_tokens"] == 207
