"""Read-only model and instance catalog snapshots."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.core.instance import ModelState


class LifecycleOperation(StrEnum):
    LOAD = "load"
    UNLOAD = "unload"


class LifecycleMetrics(BaseModel):
    """Best-effort process-level measurements around one lifecycle operation."""

    model_config = ConfigDict(frozen=True)

    operation: LifecycleOperation
    duration_ms: float = Field(ge=0)
    process_rss_before_bytes: int | None = Field(default=None, ge=0)
    process_rss_after_bytes: int | None = Field(default=None, ge=0)
    process_rss_change_bytes: int | None = None
    memory_allocated_bytes: int | None = Field(default=None, ge=0)
    memory_released_bytes: int | None = Field(default=None, ge=0)
    warmup_included: bool = False
    measured_at: datetime


class InstanceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    revision: str
    variant: str
    runtime: str
    state: ModelState
    created_at: datetime
    reference_count: int = Field(ge=0)
    load_metrics: LifecycleMetrics | None = None


class UnloadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_id: str
    revision: str
    variant: str
    runtime: str
    state: ModelState
    metrics: LifecycleMetrics


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


class VariantSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    description: str
    metadata: dict[str, str | int | float | bool]
    default: bool
    instance_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)


class ModelSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    variants: tuple[VariantSummary, ...]
    default_variant: str
    name: str
    description: str
    family: str
    tags: tuple[str, ...]
    capabilities: tuple[str, ...]
    license: str | None = None
    source_url: str | None = None
    runtimes: tuple[RuntimeSummary, ...]
    default_runtime: str | None
    instantiated: bool
    instance_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    instances_by_state: dict[str, int]
    instances_by_runtime: dict[str, int]
    instances_by_variant: dict[str, int]
    instances: tuple[InstanceSnapshot, ...]


class ModelCatalogSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    models: tuple[ModelSummary, ...]
