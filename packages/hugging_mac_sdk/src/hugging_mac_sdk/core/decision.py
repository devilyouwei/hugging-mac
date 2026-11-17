"""Shared decision dispatch, capability limits, and per-question result validation."""

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from pydantic import TypeAdapter, ValidationError

from hugging_mac_sdk.schemas.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionError,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionSupport,
    DecisionTimings,
    DecisionUsage,
    InvalidDecisionQuestion,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

_ANSWER: TypeAdapter[DecisionAnswer] = TypeAdapter(DecisionAnswer)


def question_error(
    question: DecisionQuestion | InvalidDecisionQuestion, support: DecisionSupport
) -> DecisionError | None:
    if isinstance(question, InvalidDecisionQuestion):
        return DecisionError(
            question_type=question.requested_type,
            error="invalid_question",
            message=question.message,
        )
    if question.type not in support.question_types:
        return DecisionError(
            question_type=question.type,
            error="unsupported_question_type",
            message=f"Unsupported decision type: {question.type}",
        )
    reason = None
    if (
        isinstance(question, ChoiceQuestion)
        and support.max_choices is not None
        and len(question.criteria) > support.max_choices
    ):
        reason = f"At most {support.max_choices} choices are supported"
    if (
        isinstance(question, ScoreQuestion)
        and support.max_score_levels is not None
        and len(question.criteria) > support.max_score_levels
    ):
        reason = f"At most {support.max_score_levels} score levels are supported"
    if isinstance(question, NoulQuestion) and question.criteria and not support.noul_criteria:
        reason = "Custom noul descriptions are not supported"
    return (
        DecisionError(question_type=question.type, error="unsupported_input", message=reason)
        if reason
        else None
    )


async def execute_decisions(
    request: DecisionRequest,
    *,
    support: DecisionSupport,
    model_id: str,
    instance_id: str,
    runtime: str,
    device: str,
    model_name: str,
    prepare: Callable[[DecisionRequest], dict[str, Any]],
    infer: Callable[..., Awaitable[dict[str, Any]]],
) -> DecisionResponse:
    started = perf_counter()
    answers: dict[str, DecisionAnswer] = {}
    accepted = {}
    media_error = (request.images and "image" not in support.modalities) or (
        request.videos and "video" not in support.modalities
    )
    for key, question in request.questions.items():
        error = question_error(question, support)
        if error is None and media_error:
            error = DecisionError(
                question_type=question.type,
                error="unsupported_input",
                message="This model does not support the requested media",
            )
        if error is not None:
            answers[key] = error
        else:
            accepted[key] = question
    inputs = prepare(request.model_copy(update={"questions": accepted})) if accepted else {}
    prepared = perf_counter()
    inference_ms = postprocess_ms = 0.0
    input_tokens = output_tokens = 0
    for key, question in accepted.items():
        call = inputs | {"questions": {key: inputs["questions"][key]}}
        before = perf_counter()
        try:
            result = await infer(**call)
        except ValueError as error:
            answers[key] = DecisionError(
                question_type=question.type, error="invalid_question", message=str(error)
            )
            inference_ms += (perf_counter() - before) * 1000
            continue
        inference_ms += (perf_counter() - before) * 1000
        before = perf_counter()
        try:
            usage = DecisionUsage.model_validate(result["usage"])
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            raw = result["answers"][key]
            if isinstance(raw, dict) and "error" in raw:
                raw = raw | {"type": "error", "question_type": question.type}
            answer = _ANSWER.validate_python(raw)
            if not isinstance(answer, DecisionError):
                if answer.type != question.type:
                    raise ValueError("Answer type differs from question type")
                if (
                    isinstance(answer, ChoiceAnswer)
                    and isinstance(question, ChoiceQuestion)
                    and set(answer.probabilities) != set(question.criteria)
                ):
                    raise ValueError("Answer labels differ from requested choices")
                if (
                    isinstance(answer, ScoreAnswer)
                    and isinstance(question, ScoreQuestion)
                    and len(answer.probabilities) != len(question.criteria)
                ):
                    raise ValueError("Answer levels differ from requested score levels")
            answers[key] = answer
        except (ValidationError, ValueError, KeyError, TypeError) as error:
            answers[key] = DecisionError(
                question_type=question.type, error="invalid_model_output", message=str(error)
            )
        postprocess_ms += (perf_counter() - before) * 1000
    return DecisionResponse(
        model_id=model_id,
        instance_id=instance_id,
        runtime=runtime,
        device=device,
        model=model_name,
        answers={key: answers[key] for key in request.questions},
        usage=DecisionUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        timings=DecisionTimings(
            preprocess_ms=(prepared - started) * 1000,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        ),
    )
