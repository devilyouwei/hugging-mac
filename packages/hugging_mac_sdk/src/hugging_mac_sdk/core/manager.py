"""In-process model instance reuse policy.

Large model worker/process isolation will sit above this class. This manager keeps
the SDK reuse semantics deterministic without owning IPC.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.schemas.catalog import InstanceSnapshot


class ReusePolicy(StrEnum):
    DEDICATED = "dedicated"
    SHARED = "shared"
    REUSE_IF_READY = "reuse-if-ready"


@dataclass(slots=True)
class _InstanceRecord:
    model_id: str
    revision: str
    runtime: str
    created_at: datetime
    reference_count: int = 0


class InstanceManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._instances: dict[str, BaseModelInstance] = {}
        self._shared: dict[tuple[object, ...], BaseModelInstance] = {}
        self._records: dict[str, _InstanceRecord] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        runtime: str | None = None,
        options: Mapping[str, object] | None = None,
        reuse: ReusePolicy = ReusePolicy.DEDICATED,
    ) -> BaseModelInstance:
        normalized = dict(options or {})
        manifest = self._registry.get(model_id, revision).manifest
        resolved_revision = manifest.revision
        selected_runtime = runtime or manifest.default_runtime or manifest.runtimes[0].name
        key = self._instance_key(model_id, resolved_revision, selected_runtime, normalized)

        async with self._lock:
            existing = self._shared.get(key)
            if existing is not None and (
                reuse is ReusePolicy.SHARED
                or (reuse is ReusePolicy.REUSE_IF_READY and existing.state is ModelState.READY)
            ):
                return existing

            instance = self._registry.create_instance(
                model_id,
                revision=resolved_revision,
                runtime=selected_runtime,
                options=normalized,
            )
            instance_id = str(instance.instance_id)
            self._instances[instance_id] = instance
            self._records[instance_id] = _InstanceRecord(
                model_id=model_id,
                revision=resolved_revision,
                runtime=selected_runtime,
                created_at=datetime.now(UTC),
            )
            if reuse is not ReusePolicy.DEDICATED:
                self._shared[key] = instance
            return instance

    async def get(self, instance_id: str) -> BaseModelInstance | None:
        async with self._lock:
            return self._instances.get(instance_id)

    async def unload(self, instance_id: str) -> bool:
        async with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance is None:
                return False
            self._records.pop(instance_id, None)
            stale_keys = [key for key, value in self._shared.items() if value is instance]
            for key in stale_keys:
                del self._shared[key]

        await instance.unload()
        return True

    async def list(self) -> tuple[BaseModelInstance, ...]:
        async with self._lock:
            return tuple(self._instances.values())

    async def retain(self, instance_id: str) -> None:
        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
            record.reference_count += 1

    async def release(self, instance_id: str) -> int:
        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
            record.reference_count = max(0, record.reference_count - 1)
            return record.reference_count

    async def snapshots(self) -> tuple[InstanceSnapshot, ...]:
        async with self._lock:
            return tuple(
                InstanceSnapshot(
                    instance_id=instance_id,
                    model_id=record.model_id,
                    revision=record.revision,
                    runtime=record.runtime,
                    state=self._instances[instance_id].state,
                    created_at=record.created_at,
                    reference_count=record.reference_count,
                )
                for instance_id, record in self._records.items()
            )

    @staticmethod
    def _instance_key(
        model_id: str,
        revision: str,
        runtime: str,
        options: Mapping[str, object],
    ) -> tuple[object, ...]:
        return (
            model_id,
            revision,
            runtime,
            tuple(sorted((key, _freeze(value)) for key, value in options.items())),
        )


def _freeze(value: Any) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
