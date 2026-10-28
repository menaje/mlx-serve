"""Behavioral tests for the System One decision API and D3 contract."""

import math

import pytest

from mlx_serve.core.decision import (
    D3DecisionModel,
    answer_from_probabilities,
    question_options,
    render_prompt,
)


def test_score_uses_ordered_levels_and_expected_value():
    question = {"type": "score", "criteria": ["routine", "soon", "urgent"]}
    assert question_options(question) == (["0", "1", "2"], ["routine", "soon", "urgent"])
    answer = answer_from_probabilities(question, [0.25, 0.25, 0.5])
    assert answer["score"] == pytest.approx(1.25)
    assert answer["legend"] == {"0": "routine", "1": "soon", "2": "urgent"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert 0 <= answer["confidence"] <= 1


def test_choice_probability_indices_are_option_order_not_token_ids():
    question = {"type": "choice", "criteria": {"first": "A", "second": "B"}}
    answer = answer_from_probabilities(question, [0.1, 0.9])
    assert answer["choice"] == "second"
    assert answer["probabilities"]["second"] == pytest.approx(0.9)
    expected_confidence = 1 + (0.1 * math.log(0.1) + 0.9 * math.log(0.9)) / math.log(2)
    assert answer["confidence"] == pytest.approx(expected_confidence)


def test_rendered_score_labels_do_not_include_numeric_indices():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[1]["content"]

    prompt = render_prompt(
        Tokenizer(),
        {"event": "a refund"},
        {"type": "score", "criteria": ["routine", "urgent"]},
        ["A", "B"],
    )
    assert "A: routine\nB: urgent" in prompt
    assert '"event": "a refund"' in prompt


def test_long_prompt_is_reported_without_invoking_model():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def encode(self, prompt, add_special_tokens=False):
            return [1, 2, 3]

    model = object.__new__(D3DecisionModel)
    model.tokenizer = Tokenizer()
    model.codes = ["A", "B"]
    model.max_tokens = 2
    result = model.system_one("state", {"x": {"type": "noul"}})
    assert result["answers"]["x"]["error"] == "max_length_exceeded"
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_systemone_endpoint_returns_model_answers(client, monkeypatch):
    class Model:
        def system_one(self, state, questions):
            assert state == "requesting a refund"
            assert questions["refund"]["type"] == "noul"
            return {
                "answers": {"refund": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 12, "output_tokens": 0},
            }

    monkeypatch.setattr(
        "mlx_serve.routers.decision.model_manager.get_decision_model", lambda name: Model()
    )
    response = client.post(
        "/v1/systemone",
        json={
            "model": "d3-lite",
            "state": "requesting a refund",
            "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "d3-lite"
    assert response.json()["answers"]["refund"]["noul"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "changes",
    [
        {"questions": {}},
        {"questions": {"bad name": {"type": "noul"}}},
        {"questions": {"bad": {"type": "choice", "criteria": {"one": "only"}}}},
        {"questions": {"bad": {"type": "score", "criteria": ["same", "same"]}}},
        {"questions": {"bad": {"type": "noul", "criteria": {"maybe": "yes"}}}},
        {"images": ["aW1hZ2U="]},
        {"videos": ["dmlkZW8="]},
    ],
)
def test_systemone_rejects_invalid_or_unimplemented_inputs(client, changes):
    payload = {"model": "d3-lite", "state": "test", "questions": {"q": {"type": "noul"}}}
    payload.update(changes)
    assert client.post("/v1/systemone", json=payload).status_code == 422


def test_systemone_returns_404_for_missing_model(client, monkeypatch):
    def missing(name):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr("mlx_serve.routers.decision.model_manager.get_decision_model", missing)
    response = client.post(
        "/v1/systemone",
        json={"model": "missing", "state": "test", "questions": {"q": {"type": "noul"}}},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "model_not_found"
