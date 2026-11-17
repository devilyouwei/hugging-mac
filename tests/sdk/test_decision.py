"""The same decision contract applies to both models and every declared runtime."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.d3 import D3_DEFINITION
from hugging_mac_sdk.models.vela import VELA_DEFINITION
from hugging_mac_sdk.schemas.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
)
from hugging_mac_sdk.schemas.detection import ImageInput
from pydantic import ValidationError


@pytest.fixture(
    params=[(d, r) for d in (D3_DEFINITION, VELA_DEFINITION) for r in d.runtime_factories],
    ids=["d3-pytorch", "d3-coreai", "vela-pytorch", "vela-coreai"],
)
def instance(request, monkeypatch):
    definition, runtime = request.param
    instance = definition.runtime_factories[runtime]({})
    engine = instance._engine
    for name, value in [("resolve", Path("fixture")), ("load", None), ("close", None)]:
        monkeypatch.setattr(engine, name, AsyncMock(return_value=value))

    async def infer(**inputs):
        answers = {}
        for key, question in inputs["questions"].items():
            kind = question["type"]
            if kind == "choice":
                labels = list(question["criteria"])
                answers[key] = {
                    "type": kind,
                    "choice": labels[0],
                    "probabilities": dict.fromkeys(labels, 1 / len(labels)),
                }
            elif kind == "noul":
                answers[key] = {"type": kind, "noul": 0.75}
            else:
                answers[key] = {
                    "type": kind,
                    "score": 0.75,
                    "probabilities": {"0": 0.25, "1": 0.75},
                    "legend": {"0": "low", "1": "high"},
                }
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}

    monkeypatch.setattr(engine, "infer", AsyncMock(side_effect=infer))
    return instance


async def test_shared_types_and_partial_errors(instance):
    await instance.load()
    try:
        capability = instance.require(DecisionMaking)
        support = capability.decision_support()
        d3 = "score" in support.question_types
        response = await capability.decide(
            DecisionRequest(
                state={"text": "hello"},
                questions={
                    "route": {"type": "choice", "criteria": ["a", "b", "c"]},
                    "yes": {"type": "noul"},
                    "rating": {"type": "score", "criteria": ["low", "high"]},
                    "custom": {"type": "noul", "criteria": {"true": "approved"}},
                    "broken": {"type": "choice", "criteria": []},
                    "unknown": {"type": "span"},
                },
            )
        )
        assert isinstance(response.answers["route"], ChoiceAnswer)
        assert set(response.answers["route"].probabilities) == {"a", "b", "c"}
        assert isinstance(response.answers["yes"], NoulAnswer)
        assert response.answers["yes"].noul == 0.75
        if d3:
            assert isinstance(response.answers["rating"], ScoreAnswer)
            assert isinstance(response.answers["custom"], NoulAnswer)
        else:
            assert response.answers["rating"].error == "unsupported_question_type"
            assert response.answers["custom"].error == "unsupported_input"
        for key in ("broken", "unknown"):
            assert response.answers[key].error == "invalid_question"
        assert list(response.answers) == ["route", "yes", "rating", "custom", "broken", "unknown"]
        assert response.usage.input_tokens == (40 if d3 else 20)
        assert instance._engine.infer.await_count == (4 if d3 else 2)
    finally:
        await instance.unload()


async def test_invalid_native_output_and_value_error_are_isolated(instance):
    await instance.load()
    instance._engine.infer.side_effect = [
        ValueError("invalid input"),
        {
            "answers": {
                "bad": {"type": "choice", "choice": "wrong", "probabilities": {"wrong": 1}}
            },
            "usage": {},
        },
        {"answers": {"ok": {"type": "noul", "noul": 0.6}}, "usage": {}},
    ]
    try:
        response = await instance.decide(
            DecisionRequest(
                state="",
                questions={
                    "invalid": {"type": "noul"},
                    "bad": {"type": "choice", "criteria": ["a"]},
                    "ok": {"type": "noul"},
                },
            )
        )
        assert response.answers["invalid"].error == "invalid_question"
        assert response.answers["bad"].error == "invalid_model_output"
        assert response.answers["ok"].noul == 0.6
    finally:
        await instance.unload()


async def test_empty_request_and_unsupported_media(instance):
    await instance.load()
    try:
        assert not (await instance.decide(DecisionRequest(state="", questions={}))).answers
        instance._engine.infer.assert_not_awaited()
        if "image" not in instance.decision_support().modalities:
            response = await instance.decide(
                DecisionRequest(
                    state="",
                    questions={"q": {"type": "noul"}},
                    images=(ImageInput(data=b"unread image"),),
                )
            )
            assert response.answers["q"].error == "unsupported_input"
            instance._engine.infer.assert_not_awaited()
    finally:
        await instance.unload()


def test_schema_roundtrip_and_validation():
    request = DecisionRequest(
        state=None,
        questions={
            "q": ChoiceQuestion(criteria={"a": None, "b": {"description": "B"}}),
            "score": ScoreQuestion(criteria=({"level": 0}, {"level": 1})),
            "bad": {"type": "unknown"},
            "invalid_tag": {"type": "invalid"},
        },
    )
    assert DecisionRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError):
        ChoiceQuestion(criteria=["duplicate", "duplicate"])
    with pytest.raises(ValidationError):
        NoulAnswer(noul=float("nan"))
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="a", probabilities={"a": 0.2})
    with pytest.raises(ValidationError):
        ScoreAnswer(score=0.2, probabilities={"0": 0.5, "1": 0.5}, legend={"0": "a", "1": "b"})
    assert DecisionError(error="invalid_question", message="bad").type == "error"
