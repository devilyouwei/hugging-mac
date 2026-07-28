"""Explicit model resource operations.

Loading a model must never download or convert artifacts. Applications call
this service deliberately, normally from a user-triggered workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import UnsupportedCapabilityError
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import ModelResourceStatus


class ModelResourceProvider(Protocol):
    async def status(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
    ) -> ModelResourceStatus: ...

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus: ...

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus: ...

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus: ...


class ModelResourceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    async def status(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> ModelResourceStatus:
        definition = self._registry.get(model_id, revision)
        selected_variant = definition.manifest.get_variant(variant).name
        return await self._provider(model_id, revision).status(selected_variant, options)

    async def download_source(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        options: Mapping[str, object] | None = None,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        definition = self._registry.get(model_id, revision)
        selected_variant = definition.manifest.get_variant(variant).name
        return await self._provider(model_id, revision).download_source(
            selected_variant,
            options,
            overwrite=overwrite,
        )

    async def convert(
        self,
        model_id: str,
        target_format: ArtifactFormat,
        *,
        revision: str | None = None,
        variant: str | None = None,
        options: Mapping[str, object] | None = None,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        definition = self._registry.get(model_id, revision)
        selected_variant = definition.manifest.get_variant(variant).name
        return await self._provider(model_id, revision).convert(
            selected_variant,
            target_format,
            options,
            overwrite=overwrite,
        )

    async def delete(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        runtime: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> ModelResourceStatus:
        """Delete local artifacts without changing the registered definition."""

        definition = self._registry.get(model_id, revision)
        selected_variant = definition.manifest.get_variant(variant).name
        return await self._provider(model_id, revision).delete(
            selected_variant,
            options,
            runtime=runtime,
        )

    def _provider(
        self,
        model_id: str,
        revision: str | None,
    ) -> ModelResourceProvider:
        definition = self._registry.get(model_id, revision)
        if definition.resource_provider is None:
            raise UnsupportedCapabilityError(
                f"Model {model_id} does not expose resource operations",
                details={"model_id": model_id},
            )
        return definition.resource_provider
