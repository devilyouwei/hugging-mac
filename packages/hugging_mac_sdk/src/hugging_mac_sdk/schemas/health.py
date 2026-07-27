from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: HealthStatus
    state: str
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
