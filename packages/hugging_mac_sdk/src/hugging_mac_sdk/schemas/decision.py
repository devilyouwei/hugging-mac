"""Shared typed contracts for decision models, independent of runtime and architecture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from hugging_mac_sdk.schemas.detection import ImageInput

DecisionType = Literal["choice", "noul", "score"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DecisionSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChoiceQuestion(DecisionSchema):
    type: Literal["choice"] = "choice"
    instructions: JsonValue = None
    criteria: dict[str, JsonValue] = Field(min_length=1)

    @field_validator("criteria", mode="before")
    @classmethod
    def normalize_choices(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            if any(not isinstance(item, str) or not item for item in value) or len(
                set(value)
            ) != len(value):
                raise ValueError("Choice labels must be distinct nonempty strings")
            return dict.fromkeys(value)
        return value

    @field_validator("criteria")
    @classmethod
    def validate_labels(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if any(not key for key in value):
            raise ValueError("Choice labels must not be empty")
        return value


class NoulQuestion(DecisionSchema):
    """The output is P(affirmative); optional descriptions use false/true keys."""

    type: Literal["noul"] = "noul"
    instructions: JsonValue = None
    criteria: dict[Literal["false", "true"], JsonValue] | None = None


class ScoreQuestion(DecisionSchema):
    type: Literal["score"] = "score"
    instructions: JsonValue = None
    criteria: tuple[JsonValue, ...] = Field(min_length=2)

    @field_validator("criteria")
    @classmethod
    def validate_levels(cls, value: tuple[JsonValue, ...]) -> tuple[JsonValue, ...]:
        levels = [
            level if isinstance(level, str) else json.dumps(level, ensure_ascii=False)
            for level in value
        ]
        if any(not level for level in levels) or len(set(levels)) != len(levels):
            raise ValueError("Score levels must be distinct nonempty descriptions")
        return value


DecisionQuestion = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")
]
_QUESTION: TypeAdapter[DecisionQuestion] = TypeAdapter(DecisionQuestion)


class InvalidDecisionQuestion(DecisionSchema):
    """A raw invalid batch item, retained so valid siblings can still run."""

    type: Literal["invalid"] = "invalid"
    requested_type: str | None = None
    message: str


class DecisionVideoFrames(DecisionSchema):
    frames: tuple[ImageInput, ...] = Field(min_length=1)


class DecisionRequest(DecisionSchema):
    state: JsonValue
    questions: dict[str, DecisionQuestion | InvalidDecisionQuestion]
    images: tuple[ImageInput, ...] = ()
    videos: tuple[Path | DecisionVideoFrames, ...] = ()

    @field_validator("questions", mode="before")
    @classmethod
    def parse_questions(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        parsed: dict[str, DecisionQuestion | InvalidDecisionQuestion] = {}
        for key, question in value.items():
            if isinstance(question, InvalidDecisionQuestion):
                parsed[key] = question
                continue
            try:
                if isinstance(question, dict) and question.get("type") == "invalid":
                    parsed[key] = InvalidDecisionQuestion.model_validate(question)
                else:
                    parsed[key] = _QUESTION.validate_python(question)
            except ValidationError as error:
                requested = question.get("type") if isinstance(question, dict) else None
                parsed[key] = InvalidDecisionQuestion(
                    requested_type=requested if isinstance(requested, str) else None,
                    message="; ".join(item["msg"] for item in error.errors(include_input=False)),
                )
        return parsed


class DecisionSupport(DecisionSchema):
    question_types: tuple[DecisionType, ...]
    modalities: tuple[Literal["text", "image", "video"], ...] = ("text",)
    max_choices: int | None = Field(default=None, ge=1)
    max_score_levels: int | None = Field(default=None, ge=2)
    noul_criteria: bool = False


class DecisionAction(DecisionSchema):
    act_probability: Probability


class ChoiceAnswer(DecisionSchema):
    action: DecisionAction | None = None
    answer_confidence: Probability | None = None
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability] = Field(min_length=1)
    confidence: Probability | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("Selected choice is missing from probabilities")
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5):
            raise ValueError("Choice probabilities must sum to one")
        return self


class NoulAnswer(DecisionSchema):
    confidence: Probability | None = None
    action: DecisionAction | None = None
    answer_confidence: Probability | None = None
    type: Literal["noul"] = "noul"
    noul: Probability


class ScoreAnswer(DecisionSchema):
    action: DecisionAction | None = None
    answer_confidence: Probability | None = None
    type: Literal["score"] = "score"
    score: float = Field(ge=0, allow_inf_nan=False)
    probabilities: dict[str, Probability] = Field(min_length=2)
    confidence: Probability | None = None
    legend: dict[str, str]

    @model_validator(mode="after")
    def validate_distribution(self) -> ScoreAnswer:
        keys = {str(index) for index in range(len(self.probabilities))}
        if set(self.probabilities) != keys or set(self.legend) != keys:
            raise ValueError("Score levels must use consecutive zero-based indices")
        expected = sum(int(key) * value for key, value in self.probabilities.items())
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5) or not math.isclose(
            self.score, expected, abs_tol=1e-5
        ):
            raise ValueError("Score must be the expected zero-based level index")
        return self


class DecisionError(DecisionSchema):
    type: Literal["error"] = "error"
    question_type: str | None = None
    error: Literal[
        "invalid_question",
        "unsupported_question_type",
        "unsupported_input",
        "max_length_exceeded",
        "invalid_model_output",
        "inference_error",
    ]
    message: str


DecisionAnswer = Annotated[
    ChoiceAnswer | NoulAnswer | ScoreAnswer | DecisionError, Field(discriminator="type")
]


class DecisionUsage(DecisionSchema):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class DecisionTimings(DecisionSchema):
    preprocess_ms: float = Field(default=0.0, ge=0.0)
    inference_ms: float = Field(default=0.0, ge=0.0)
    postprocess_ms: float = Field(default=0.0, ge=0.0)


class DecisionResponse(DecisionSchema):
    model_id: str
    instance_id: str
    runtime: str
    device: str
    model: str
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage
    timings: DecisionTimings = DecisionTimings()
