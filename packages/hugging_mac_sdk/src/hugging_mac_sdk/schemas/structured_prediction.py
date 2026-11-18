"""Lossless JSON boundary for models with upstream structured task routing."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue


class StructuredPredictionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: JsonValue
    questions: JsonValue
    method: Literal["system_one", "predict"] = "system_one"
    model: JsonValue = None


class StructuredPredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: JsonValue
