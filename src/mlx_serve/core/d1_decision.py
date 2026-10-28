"""d1-omni text decisions using the official mlx-vlm implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_serve.config import settings
from mlx_serve.core.decision import answer_from_probabilities


class DecisionInputError(ValueError):
    """A decision cannot be evaluated without silently truncating its input."""


class D1DecisionModel:
    """Run the upstream d1 model while rejecting inputs it would truncate."""

    def __init__(self, model_dir: Path):
        import mlx.core as mx
        from mlx_vlm import load

        self.stream = mx.new_stream(mx.default_device())
        with mx.stream(self.stream):
            self.model, self.processor = load(str(model_dir), trust_remote_code=False, strict=True)
        if getattr(self.model, "model_type", None) != "d1_omni":
            raise ValueError("Checkpoint is not a supported d1-omni model")
        self.max_tokens = min(settings.decision_max_input_tokens, self.model.config.max_length)
        self.model.config.max_length = self.max_tokens

    def _check_input(self, state: Any, question: dict[str, Any]) -> None:
        from mlx_vlm.models.d1_omni.prompt import as_question, escape, render_options, serialize

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        parsed = as_question(question)

        def token_count(text: str) -> int:
            return len(tokenizer(escape(text), add_special_tokens=False)["input_ids"])

        options = render_options(parsed)
        budget = max(96, min(len(options) * 24 + 32, self.max_tokens // 2))
        if token_count(parsed[1]) + 1 > max(16, budget):
            raise DecisionInputError("Question instructions exceed the d1 input budget")
        per_option = max(2, (budget - 3 * len(options)) // len(options))
        option_lengths = [token_count(" " + option) for option in options]
        if any(length > per_option for length in option_lengths):
            raise DecisionInputError("A choice or score description exceeds the d1 input budget")
        question_tokens = 1 + token_count(parsed[1]) + sum(3 + n for n in option_lengths) + 1
        state_room = max(0, self.max_tokens - question_tokens - 2)
        if token_count(serialize(state)) > state_room:
            raise DecisionInputError(
                f"State exceeds the d1 input limit of {self.max_tokens} tokens"
            )
        if question_tokens + 2 > self.max_tokens:
            raise DecisionInputError(
                f"Question exceeds the d1 input limit of {self.max_tokens} tokens"
            )

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_vlm import predict

        normalized = {}
        for name, question in questions.items():
            question = dict(question)
            question["instructions"] = (
                question.get("instructions") or "Choose the best matching option."
            )
            self._check_input(state, question)
            normalized[name] = question
        with mx.stream(self.stream):
            result = predict(self.model, self.processor, state, normalized)
        return {"answers": result["answers"], "usage": result["usage"]}

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        result = self.decide(state, questions)
        answers = {}
        for name, answer in result["answers"].items():
            question = dict(questions[name])
            if question["type"] == "bool":
                question["type"] = "noul"
            if question["type"] == "noul":
                p = answer["probability"]
                probabilities = [1 - p, p]
            else:
                probabilities = list(answer["probabilities"].values())
            answers[name] = answer_from_probabilities(question, probabilities)
        return {"answers": answers, "usage": result["usage"]}
