"""In-process model instance reuse policy.

Large model worker/process isolation will sit above this class. This manager keeps
the SDK reuse semantics deterministic without owning IPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import psutil

from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.core.runtime_policy import RuntimePolicy
from hugging_mac_sdk.errors import (
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.schemas.catalog import (
    InstanceSnapshot,
    LifecycleMetrics,
    LifecycleOperation,
    UnloadResult,
)


class ReusePolicy(StrEnum):
    DEDICATED = "dedicated"
    SHARED = "shared"
    REUSE_IF_READY = "reuse-if-ready"


@dataclass(slots=True)
class _InstanceRecord:
    model_id: str
    revision: str
    variant: str
    runtime: str
    created_at: datetime
    options: dict[str, object]
    reference_count: int = 0
    load_metrics: LifecycleMetrics | None = None


class InstanceManager:
    def __init__(
        self,
        registry: ModelRegistry,
        runtime_policy: RuntimePolicy | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_policy = runtime_policy or RuntimePolicy()
        self._instances: dict[str, BaseModelInstance] = {}
        self._shared: dict[tuple[object, ...], BaseModelInstance] = {}
        self._records: dict[str, _InstanceRecord] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        runtime: str | None = None,
        device: str | None = None,
        options: Mapping[str, object] | None = None,
        reuse: ReusePolicy = ReusePolicy.DEDICATED,
    ) -> BaseModelInstance:
        normalized = dict(options or {})
        manifest = self._registry.get(model_id, revision).manifest
        resolved_revision = manifest.revision
        option_variant = normalized.get("variant")
        if variant is not None and option_variant not in {None, variant}:
            raise ResourceNotFoundError(
                "Conflicting variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        selected_variant = manifest.get_variant(
            variant or (str(option_variant) if option_variant is not None else None)
        ).name
        normalized["variant"] = selected_variant
        selected_runtime = self._runtime_policy.resolve(manifest, runtime)
        runtime_spec = next(
            item for item in manifest.runtimes if item.name == selected_runtime
        )
        option_device = normalized.get("device")
        if device is not None and option_device not in {None, device}:
            raise UnsupportedRuntimeError(
                "Conflicting device values were provided",
                details={"device": device, "options_device": option_device},
            )
        if device is not None:
            selected_device = self._runtime_policy.select_device(runtime_spec, device)
            if selected_device is not None:
                normalized["device"] = selected_device
        key = self._instance_key(
            model_id,
            resolved_revision,
            selected_variant,
            selected_runtime,
            normalized,
        )

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
                variant=selected_variant,
                runtime=selected_runtime,
                options=normalized,
            )
            instance_id = str(instance.instance_id)
            self._instances[instance_id] = instance
            self._records[instance_id] = _InstanceRecord(
                model_id=model_id,
                revision=resolved_revision,
                variant=selected_variant,
                runtime=selected_runtime,
                created_at=datetime.now(UTC),
                options=normalized,
            )
            if reuse is not ReusePolicy.DEDICATED:
                self._shared[key] = instance
            return instance

    async def load(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        runtime: str | None = "auto",
        device: str | None = None,
        options: Mapping[str, object] | None = None,
        reuse: ReusePolicy = ReusePolicy.DEDICATED,
        warmup: bool = False,
    ) -> BaseModelInstance:
        """Create and load one instance through the complete manager pipeline."""

        instance = await self.create(
            model_id,
            revision=revision,
            variant=variant,
            runtime=runtime,
            device=device,
            options=options,
            reuse=reuse,
        )
        try:
            return await self.ensure_loaded(
                str(instance.instance_id),
                warmup=warmup,
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await self.unload(str(instance.instance_id))
            raise

    async def ensure_loaded(
        self,
        instance_id: str,
        *,
        warmup: bool = False,
    ) -> BaseModelInstance:
        """Load a managed instance and persist metrics for its first real load."""

        instance = await self.require(instance_id)
        if instance.state is ModelState.READY:
            if warmup:
                await instance.warmup()
            return instance
        rss_before = _process_rss()
        started = time.perf_counter_ns()
        await instance.load()
        if warmup:
            await instance.warmup()
        metrics = _lifecycle_metrics(
            LifecycleOperation.LOAD,
            started,
            rss_before,
            warmup_included=warmup,
        )
        async with self._lock:
            record = self._records.get(str(instance.instance_id))
            if record is not None:
                record.load_metrics = metrics
        return instance

    async def get(self, instance_id: str) -> BaseModelInstance | None:
        async with self._lock:
            return self._instances.get(instance_id)

    async def require(self, instance_id: str) -> BaseModelInstance:
        instance = await self.get(instance_id)
        if instance is None:
            raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
        return instance

    async def switch_runtime(
        self,
        instance_id: str,
        runtime: str,
        *,
        device: str | None = None,
        options: Mapping[str, object] | None = None,
        warmup: bool = False,
    ) -> BaseModelInstance:
        """Replace an unreferenced instance with a newly loaded runtime instance."""

        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
            if record.reference_count:
                raise UnsupportedRuntimeError(
                    "Cannot switch a retained model instance",
                    details={
                        "instance_id": instance_id,
                        "reference_count": record.reference_count,
                    },
                )
            model_id = record.model_id
            revision = record.revision
            original_record = record
            inherited_options = dict(record.options)
        inherited_options.pop("runtime", None)
        inherited_options.pop("device", None)
        inherited_options.update(options or {})
        replacement = await self.load(
            model_id,
            revision=revision,
            runtime=runtime,
            device=device,
            options=inherited_options,
            warmup=warmup,
        )
        async with self._lock:
            current_record = self._records.get(instance_id)
            can_replace = (
                current_record is original_record
                and current_record.reference_count == 0
            )
        if not can_replace:
            await self.unload(str(replacement.instance_id))
            raise UnsupportedRuntimeError(
                "Instance ownership changed while switching runtime",
                details={"instance_id": instance_id},
            )
        await self.unload(instance_id)
        return replacement

    async def switch_variant(
        self,
        instance_id: str,
        variant: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        options: Mapping[str, object] | None = None,
        warmup: bool = False,
    ) -> BaseModelInstance:
        """Replace an unreferenced instance with another weight variant."""

        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
            if record.reference_count:
                raise UnsupportedRuntimeError(
                    "Cannot switch a retained model instance",
                    details={
                        "instance_id": instance_id,
                        "reference_count": record.reference_count,
                    },
                )
            model_id = record.model_id
            revision = record.revision
            selected_runtime = runtime or record.runtime
            original_record = record
            inherited_options = dict(record.options)
        inherited_options.pop("runtime", None)
        inherited_options.pop("variant", None)
        inherited_options.pop("device", None)
        inherited_options.update(options or {})
        replacement = await self.load(
            model_id,
            revision=revision,
            variant=variant,
            runtime=selected_runtime,
            device=device,
            options=inherited_options,
            warmup=warmup,
        )
        async with self._lock:
            current_record = self._records.get(instance_id)
            can_replace = (
                current_record is original_record
                and current_record.reference_count == 0
            )
        if not can_replace:
            await self.unload(str(replacement.instance_id))
            raise UnsupportedRuntimeError(
                "Instance ownership changed while switching variant",
                details={"instance_id": instance_id},
            )
        await self.unload(instance_id)
        return replacement

    async def unload(self, instance_id: str) -> bool:
        return await self.unload_with_metrics(instance_id) is not None

    async def unload_with_metrics(
        self,
        instance_id: str,
        *,
        force: bool = False,
    ) -> UnloadResult | None:
        """Unload an instance and return process-level release measurements."""

        async with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance is None:
                return None
            record = self._records[instance_id]
            if record.reference_count and not force:
                self._instances[instance_id] = instance
                raise UnsupportedRuntimeError(
                    "Cannot unload a retained model instance",
                    details={
                        "instance_id": instance_id,
                        "reference_count": record.reference_count,
                    },
                )
            self._records.pop(instance_id)
            stale_keys = [key for key, value in self._shared.items() if value is instance]
            for key in stale_keys:
                del self._shared[key]

        rss_before = _process_rss()
        started = time.perf_counter_ns()
        await instance.unload()
        metrics = _lifecycle_metrics(
            LifecycleOperation.UNLOAD,
            started,
            rss_before,
        )
        return UnloadResult(
            instance_id=instance_id,
            model_id=record.model_id,
            revision=record.revision,
            variant=record.variant,
            runtime=record.runtime,
            state=instance.state,
            metrics=metrics,
        )

    async def list(self) -> tuple[BaseModelInstance, ...]:
        async with self._lock:
            return tuple(self._instances.values())

    async def unload_all(self, *, force: bool = False) -> None:
        """Unload every managed instance, continuing after individual failures."""

        async with self._lock:
            instance_ids = tuple(self._instances)
        failures: list[Exception] = []
        for instance_id in instance_ids:
            try:
                await self.unload_with_metrics(instance_id, force=force)
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("One or more model instances failed to unload", failures)

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
                    variant=record.variant,
                    runtime=record.runtime,
                    state=self._instances[instance_id].state,
                    created_at=record.created_at,
                    reference_count=record.reference_count,
                    load_metrics=record.load_metrics,
                )
                for instance_id, record in self._records.items()
            )

    async def snapshot(self, instance_id: str) -> InstanceSnapshot:
        async with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
            return InstanceSnapshot(
                instance_id=instance_id,
                model_id=record.model_id,
                revision=record.revision,
                variant=record.variant,
                runtime=record.runtime,
                state=self._instances[instance_id].state,
                created_at=record.created_at,
                reference_count=record.reference_count,
                load_metrics=record.load_metrics,
            )

    @staticmethod
    def _instance_key(
        model_id: str,
        revision: str,
        variant: str,
        runtime: str,
        options: Mapping[str, object],
    ) -> tuple[object, ...]:
        return (
            model_id,
            revision,
            variant,
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


def _process_rss() -> int | None:
    try:
        return int(psutil.Process().memory_info().rss)
    except (OSError, psutil.Error):
        return None


def _lifecycle_metrics(
    operation: LifecycleOperation,
    started_ns: int,
    rss_before: int | None,
    *,
    warmup_included: bool = False,
) -> LifecycleMetrics:
    duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    rss_after = _process_rss()
    change = (
        rss_after - rss_before
        if rss_before is not None and rss_after is not None
        else None
    )
    return LifecycleMetrics(
        operation=operation,
        duration_ms=duration_ms,
        process_rss_before_bytes=rss_before,
        process_rss_after_bytes=rss_after,
        process_rss_change_bytes=change,
        memory_allocated_bytes=max(change, 0) if change is not None else None,
        memory_released_bytes=max(-change, 0) if change is not None else None,
        warmup_included=warmup_included,
        measured_at=datetime.now(UTC),
    )
