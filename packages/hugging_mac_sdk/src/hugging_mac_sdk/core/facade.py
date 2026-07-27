"""Application-facing SDK facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from hugging_mac_sdk.core.catalog import ModelCatalogService
from hugging_mac_sdk.core.instance import BaseModelInstance
from hugging_mac_sdk.core.manager import InstanceManager, ReusePolicy
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.core.resources import ModelResourceService
from hugging_mac_sdk.core.runtime_policy import RuntimePolicy

CapabilityT = TypeVar("CapabilityT")


class ModelHandle:
    def __init__(
        self,
        instance: BaseModelInstance,
        manager: InstanceManager,
        *,
        unload_on_close: bool,
    ) -> None:
        self._instance = instance
        self._manager = manager
        self._unload_on_close = unload_on_close
        self._closed = False

    @property
    def instance(self) -> BaseModelInstance:
        return self._instance

    def require(self, capability: type[CapabilityT]) -> CapabilityT:
        return self._instance.require(capability)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        instance_id = str(self._instance.instance_id)
        reference_count = await self._manager.release(instance_id)
        if self._unload_on_close and reference_count == 0:
            await self._manager.unload(instance_id)

    async def __aenter__(self) -> ModelHandle:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class ModelSdk:
    """Small facade used by application services."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        instances: InstanceManager | None = None,
        runtime_policy: RuntimePolicy | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.runtime_policy = runtime_policy or RuntimePolicy()
        self.instances = instances or InstanceManager(self.registry)
        self.resources = ModelResourceService(self.registry)
        self.catalog = ModelCatalogService(
            self.registry,
            self.instances,
            self.runtime_policy,
        )

    async def acquire(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        runtime: str | None = None,
        options: Mapping[str, object] | None = None,
        reuse: ReusePolicy = ReusePolicy.SHARED,
        load: bool = True,
    ) -> ModelHandle:
        manifest = self.registry.get(model_id, revision).manifest
        selected_runtime = runtime or self.runtime_policy.select(manifest)
        instance = await self.instances.create(
            model_id,
            revision=manifest.revision,
            runtime=selected_runtime,
            options=options,
            reuse=reuse,
        )
        instance_id = str(instance.instance_id)
        await self.instances.retain(instance_id)
        try:
            if load:
                await instance.load()
        except BaseException:
            await self.instances.release(instance_id)
            if reuse is ReusePolicy.DEDICATED:
                await self.instances.unload(instance_id)
            raise
        return ModelHandle(
            instance,
            self.instances,
            unload_on_close=reuse is ReusePolicy.DEDICATED,
        )
