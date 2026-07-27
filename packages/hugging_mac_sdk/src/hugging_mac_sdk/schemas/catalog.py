"""Read-only model and instance catalog snapshots."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.core.instance import ModelState


class InstanceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    revision: str
    runtime: str
    state: ModelState
    created_at: datetime
    reference_count: int = Field(ge=0)


class RuntimeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    default: bool
    devices: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    unavailable_reason: str | None = None
    instance_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)


class ModelSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    name: str
    description: str
    family: str
    tags: tuple[str, ...]
    capabilities: tuple[str, ...]
    runtimes: tuple[RuntimeSummary, ...]
    default_runtime: str | None
    instantiated: bool
    instance_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    instances_by_state: dict[str, int]
    instances_by_runtime: dict[str, int]
    instances: tuple[InstanceSnapshot, ...]


class ModelCatalogSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    models: tuple[ModelSummary, ...]
