"""Explicit model resource operations.

Loading a model must never download or convert artifacts. Applications call
this service deliberately, normally from a user-triggered workflow.
"""

from __future__ import annotations

import glob
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedCapabilityError
from hugging_mac_sdk.resources.downloader import ProgressCallback, ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.artifact import ArtifactKind, ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
)


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
        status = await self._provider(model_id, revision).status(selected_variant, options)
        return self._with_shared_status(definition, status, options)

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
        await self.ensure_shared_artifacts(
            model_id,
            revision=revision,
            variant=selected_variant,
            options=options,
            overwrite=overwrite,
        )
        status = await self._provider(model_id, revision).download_source(
            selected_variant,
            options,
            overwrite=overwrite,
        )
        return self._with_shared_status(definition, status, options)

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
        await self.ensure_shared_artifacts(
            model_id,
            revision=revision,
            variant=selected_variant,
            target_format=target_format.value,
            options=options,
            overwrite=False,
        )
        status = await self._provider(model_id, revision).convert(
            selected_variant,
            target_format,
            options,
            overwrite=overwrite,
        )
        return self._with_shared_status(definition, status, options)

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
        status = await self._provider(model_id, revision).delete(
            selected_variant,
            options,
            runtime=runtime,
        )
        return self._with_shared_status(definition, status, options)

    async def ensure_shared_artifacts(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        variant: str | None = None,
        runtime: str | None = None,
        artifact_id: str | None = None,
        target_format: str | None = None,
        options: Mapping[str, object] | None = None,
        overwrite: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Install shares required by matching artifacts of the selected variant."""

        definition = self._registry.get(model_id, revision)
        model_home = self._model_home(options)
        downloader = ResourceDownloader()
        token_value = (options or {}).get("hf_token")
        token = str(token_value) if token_value is not None else None
        for artifact in definition.required_shared_artifacts(
            variant=variant,
            runtime=runtime,
            artifact_id=artifact_id,
            target_format=target_format,
        ):
            path = artifact.resolve(model_home)
            if not overwrite and artifact_available(artifact, path):
                continue
            assert artifact.source is not None
            await downloader.download(
                artifact.source,
                path,
                overwrite=overwrite,
                token=token,
                progress=progress,
            )

    async def download_shared_artifact(
        self,
        model_id: str,
        artifact_id: str,
        *,
        revision: str | None = None,
        options: Mapping[str, object] | None = None,
        overwrite: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Install one explicitly selected model-wide artifact."""

        definition = self._registry.get(model_id, revision)
        artifact = next(
            (item for item in definition.shared_artifacts if item.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            raise ResourceNotFoundError(
                f"Shared artifact is not declared: {artifact_id}",
                details={"model_id": model_id, "artifact_id": artifact_id},
            )
        path = artifact.resolve(self._model_home(options))
        if not overwrite and artifact_available(artifact, path):
            return
        token_value = (options or {}).get("hf_token")
        assert artifact.source is not None
        await ResourceDownloader().download(
            artifact.source,
            path,
            overwrite=overwrite,
            token=str(token_value) if token_value is not None else None,
            progress=progress,
        )

    def _with_shared_status(
        self,
        definition: ModelDefinition,
        status: ModelResourceStatus,
        options: Mapping[str, object] | None,
    ) -> ModelResourceStatus:
        declared_shared_ids = {artifact.artifact_id for artifact in definition.shared_artifacts}
        scoped_statuses: list[ModelArtifactStatus] = []
        required_share_ids: set[str] = set()
        for item in status.artifacts:
            if item.shared or item.artifact_id in declared_shared_ids:
                continue
            declared = next(
                (
                    artifact
                    for artifact in definition.artifacts
                    if not artifact.shared
                    and artifact.variant == status.variant
                    and artifact.runtime == item.runtime
                    and artifact.artifact_id == item.artifact_id
                ),
                None,
            )
            required_shares = declared.required_shares if declared is not None else ()
            required_share_ids.update(required_shares)
            scoped_statuses.append(item.model_copy(update={"required_shares": required_shares}))
        shared_artifacts = tuple(
            artifact
            for artifact in definition.shared_artifacts
            if artifact.artifact_id in required_share_ids
        )
        artifacts = tuple(scoped_statuses) + tuple(
            self._shared_status(artifact, options) for artifact in shared_artifacts
        )
        return ModelResourceStatus(
            model_id=status.model_id,
            revision=status.revision,
            variant=status.variant,
            artifacts=artifacts,
        )

    def _shared_status(
        self,
        artifact: ModelArtifact,
        options: Mapping[str, object] | None,
    ) -> ModelArtifactStatus:
        path = artifact.resolve(self._model_home(options))
        available = artifact_available(artifact, path)
        return ModelArtifactStatus(
            artifact_id=artifact.artifact_id,
            format=artifact.format.value,
            runtime=None,
            shared=True,
            available=available,
            size_bytes=(directory_size(path) if path.is_dir() else path.stat().st_size)
            if available
            else None,
        )

    def _model_home(self, options: Mapping[str, object] | None) -> Path:
        configured = (options or {}).get("model_home")
        return (
            Path(cast(str | Path, configured))
            if configured is not None
            else self._registry.storage_root
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


def artifact_available(artifact: ModelArtifact, path: Path) -> bool:
    """Return whether a declared artifact is complete enough to use."""
    if artifact.kind is ArtifactKind.FILE:
        return path.is_file()
    if not path.is_dir():
        return False
    source = artifact.source
    if not isinstance(source, HuggingFaceSource):
        return True
    patterns = source.allow_patterns
    if source.strip_prefix is not None:
        prefix = f"{source.strip_prefix.as_posix().rstrip('/')}/"
        patterns = tuple(pattern.removeprefix(prefix) for pattern in patterns)
    for pattern in patterns:
        if glob.has_magic(pattern):
            if not any(path.glob(pattern)):
                return False
        elif not (path / pattern).exists():
            return False
    return True
