"""Platform API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResponseMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    schema_version: str = "1"


class ApiResponse[DataT](BaseModel):
    model_config = ConfigDict(frozen=True)

    data: DataT
    meta: ResponseMeta


class ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    retryable: bool = False
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: ErrorBody
