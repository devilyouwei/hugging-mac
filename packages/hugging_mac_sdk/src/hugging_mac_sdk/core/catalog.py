"""Aggregate model definitions and live instance snapshots."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.manager import InstanceManager
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.core.runtime_policy import RuntimePolicy
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.schemas.catalog import (
    ModelCatalogSnapshot,
    ModelSummary,
    RuntimeSummary,
)
from hugging_mac_sdk.schemas.manifest import ModelManifest


class ModelCatalogService:
    def __init__(
        self,
        registry: ModelRegistry,
        instances: InstanceManager,
        runtime_policy: RuntimePolicy,
    ) -> None:
        self._registry = registry
        self._instances = instances
        self._runtime_policy = runtime_policy

    async def snapshot(self) -> ModelCatalogSnapshot:
        instance_snapshots = await self._instances.snapshots()
        summaries: list[ModelSummary] = []
        for definition in self._registry.list():
            manifest = definition.manifest
            model_instances = tuple(
                item
                for item in instance_snapshots
                if item.model_id == manifest.model_id and item.revision == manifest.revision
            )
            state_counts = Counter(item.state.value for item in model_instances)
            runtime_counts = Counter(item.runtime for item in model_instances)
            default_runtime = self._select_default_runtime(manifest)
            runtimes = tuple(
                RuntimeSummary(
                    name=runtime.name,
                    available=availability.available,
                    default=runtime.name == default_runtime,
                    devices=runtime.devices,
                    dtypes=runtime.dtypes,
                    quantizations=runtime.quantizations,
                    unavailable_reason=availability.reason,
                    instance_count=runtime_counts[runtime.name],
                    ready_count=sum(
                        1
                        for item in model_instances
                        if item.runtime == runtime.name and item.state is ModelState.READY
                    ),
                )
                for runtime in manifest.runtimes
                if (availability := self._runtime_policy.availability(runtime))
            )
            summaries.append(
                ModelSummary(
                    model_id=manifest.model_id,
                    revision=manifest.revision,
                    name=manifest.display_name,
                    description=manifest.description,
                    family=manifest.family,
                    tags=tuple(sorted(manifest.tags)),
                    capabilities=tuple(sorted(manifest.capabilities)),
                    runtimes=runtimes,
                    default_runtime=default_runtime,
                    instantiated=bool(model_instances),
                    instance_count=len(model_instances),
                    ready_count=state_counts[ModelState.READY.value],
                    instances_by_state=dict(sorted(state_counts.items())),
                    instances_by_runtime=dict(sorted(runtime_counts.items())),
                    instances=model_instances,
                )
            )
        return ModelCatalogSnapshot(
            generated_at=datetime.now(UTC),
            models=tuple(summaries),
        )

    def _select_default_runtime(self, manifest: ModelManifest) -> str | None:
        try:
            return self._runtime_policy.select(manifest)
        except UnsupportedRuntimeError:
            # Catalog discovery must remain available even when this machine
            # cannot execute any of the model's declared runtimes.
            return None
