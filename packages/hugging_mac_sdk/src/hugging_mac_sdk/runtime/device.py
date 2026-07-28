"""Runtime-neutral execution device descriptors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DeviceKind(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    MPS = "mps"
    ANE = "ane"
    REMOTE = "remote"
    OTHER = "other"


class DeviceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: DeviceKind = DeviceKind.OTHER
    available: bool = True
    reason: str | None = None
