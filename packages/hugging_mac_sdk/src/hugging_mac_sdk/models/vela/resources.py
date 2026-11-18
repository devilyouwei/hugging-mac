"""Scoped Vela resource operations and views over core-managed shared artifacts."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from hugging_mac_sdk.core.config import ModelPackageConfig
from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import VelaInstanceConfig
from .converter import VelaCoreAIConverter


class VelaResourceResolver:
    def __init__(
        self,
        config: VelaInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        shared_artifacts: tuple[ModelArtifact, ...] = (),
    ) -> None:
        self._config = config
        self._manifest, self._artifact = manifest, artifact
        self._shares = tuple(
            share for share in shared_artifacts if share.artifact_id in artifact.required_shares
        )
        if {share.artifact_id for share in self._shares} != set(artifact.required_shares):
            raise ValueError("Vela resolver requires every declared shared artifact")

    @property
    def path(self) -> Path:
        override = self._config.artifact_path
        if self._artifact.runtime == "pytorch":
            override = self._config.source_path or override
        return (
            override.expanduser().resolve()
            if override is not None
            else self._artifact.resolve(self._config.model_home)
        )

    def shared_paths(self) -> tuple[Path, ...]:
        # Explicit complete local bundles remain supported, but managed artifacts
        # always consume canonical shared directories, never embedded copies.
        external = self._config.artifact_path is not None or (
            self._artifact.runtime == "pytorch" and self._config.source_path is not None
        )
        return tuple(
            self.path
            if external and artifact_available(share, self.path)
            else share.resolve(self._config.model_home)
            for share in self._shares
        )

    async def resolve(self) -> Path:
        if not artifact_available(self._artifact, self.path):
            raise ResourceNotFoundError("Vela checkpoint is incomplete")
        for share, path in zip(self._shares, self.shared_paths(), strict=True):
            if not artifact_available(share, path):
                raise ResourceNotFoundError(
                    "Vela shared resource is incomplete", details={"artifact_id": share.artifact_id}
                )
        return self.path

    @contextmanager
    def view(self) -> Iterator[Path]:
        with merged_directory_view(self.path, self.shared_paths()) as root:
            yield root

    def status(self) -> ModelResourceStatus:
        available = artifact_available(self._artifact, self.path)
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id=self._artifact.artifact_id,
                    format=self._artifact.format.value,
                    runtime=self._artifact.runtime,
                    provisioning="convert" if self._artifact.convert else "download",
                    required_shares=self._artifact.required_shares,
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
            ),
        )

    async def delete(self) -> None:
        try:
            target = self.path.expanduser().resolve(strict=False)
            expected = self._artifact.resolve(self._config.model_home)
        except ValueError as error:
            raise ResourceIntegrityError(
                "Vela artifact escapes model storage", cause=error
            ) from error
        if target != expected:
            raise ResourceIntegrityError("Refusing to delete an externally supplied Vela path")
        model_root = (
            self._config.model_home.expanduser()
            .resolve(strict=False)
            .joinpath(*self._manifest.model_id.split("/"))
        )
        if not target.is_relative_to(model_root):
            raise ResourceIntegrityError("Refusing to delete Vela resources outside model root")
        if target.exists():
            await asyncio.to_thread(shutil.rmtree, target)


class VelaResourceProvider:
    def __init__(self, package: ModelPackageConfig, converter: VelaCoreAIConverter) -> None:
        self._package, self._converter = package, converter

    def _resolver(
        self,
        variant: str,
        runtime: str,
        options: Mapping[str, object] | None,
    ) -> VelaResourceResolver:
        artifact = self._package.get_artifact(shared=False, variant=variant, runtime=runtime)
        normalized = dict(options or {})
        selected = normalized.pop("runtime", self._package.manifest.default_runtime)
        if selected != runtime:
            normalized.pop("artifact_path", None)
        config = VelaInstanceConfig.model_validate(
            normalized | {"variant": variant, "runtime": runtime}
        )
        return VelaResourceResolver(
            config,
            manifest=self._package.manifest,
            artifact=artifact,
            shared_artifacts=tuple(a for a in self._package.artifacts if a.shared),
        )

    async def status(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
    ) -> ModelResourceStatus:
        artifacts = tuple(
            item
            for artifact in self._package.get_artifacts(variant=variant)
            for item in self._resolver(variant, str(artifact.runtime), options).status().artifacts
        )
        shares = self._package.get_artifacts(variant=variant, include_shared=True)
        config = VelaInstanceConfig.model_validate(dict(options or {}) | {"variant": variant})
        artifacts += tuple(
            ModelArtifactStatus(
                artifact_id=share.artifact_id,
                format=share.format.value,
                shared=True,
                available=artifact_available(share, share.resolve(config.model_home)),
                size_bytes=directory_size(share.resolve(config.model_home))
                if artifact_available(share, share.resolve(config.model_home))
                else None,
            )
            for share in shares
            if share.shared
        )
        return ModelResourceStatus(
            model_id=self._package.manifest.model_id,
            revision=self._package.manifest.revision,
            variant=variant,
            artifacts=artifacts,
        )

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        runtime = str((options or {}).get("runtime", "pytorch"))
        artifact = self._package.get_artifact(shared=False, variant=variant, runtime=runtime)
        if artifact.source is None:
            raise UnsupportedRuntimeError("No download source is declared for this Vela artifact")
        resolver = self._resolver(variant, runtime, options)
        if overwrite or not resolver.status().artifacts[0].available:
            await ResourceDownloader(timeout=3600).download(
                artifact.source,
                resolver.path,
                overwrite=overwrite,
                token=resolver._config.hf_token,
            )
        await resolver.resolve()
        return await self.status(variant, options)

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        if target_format is not ArtifactFormat.COREAI:
            raise UnsupportedRuntimeError(f"Vela cannot convert to {target_format}")
        target = self._package.get_artifact(shared=False, variant=variant, runtime="coreai")
        if not target.convert:
            raise UnsupportedRuntimeError("This Vela variant uses a prebuilt Core AI artifact")
        source = self._package.get_artifact(shared=False, variant=variant, runtime="pytorch")
        resolver = self._resolver(variant, "pytorch", options)
        path = await resolver.resolve()
        assert source.source is not None
        with resolver.view() as view:
            await self._converter.convert(
                ConversionRequest(
                    source=ResolvedResource(
                        path=view, source=source.source, size_bytes=directory_size(path)
                    ),
                    source_format=source.format,
                    target_format=target_format,
                    output_path=target.resolve(resolver._config.model_home),
                    model_id=self._package.manifest.model_id,
                    variant=variant,
                    overwrite=overwrite,
                )
            )
        return await self.status(variant, options)

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime is not None and runtime not in {r.name for r in self._package.manifest.runtimes}:
            raise UnsupportedRuntimeError(f"Vela does not support runtime {runtime}")
        for artifact in self._package.get_artifacts(variant=variant):
            if runtime is None or artifact.runtime == runtime:
                await self._resolver(variant, str(artifact.runtime), options).delete()
        return await self.status(variant, options)
