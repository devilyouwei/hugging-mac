"""Resource lifecycle for the prebuilt DeepFilterNet3 artifact."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path

from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import DeepFilterNet3InstanceConfig


class DeepFilterNet3ResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: DeepFilterNet3InstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._manifest, self._artifact = manifest, artifact
        self._downloader = downloader or ResourceDownloader()

    def root(self) -> Path:
        if self._config.artifact_path is not None:
            return self._config.artifact_path
        return self._artifact.resolve(self._config.model_home)

    async def resolve(self) -> ResolvedResource:
        root = self.root()
        missing = [name for name in self._artifact.required_files if not (root / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "DeepFilterNet3 Core ML artifact is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=root,
            source=self._source,
            digest=directory_sha256(root),
            size_bytes=directory_size(root),
        )

    async def download(self, *, overwrite: bool = False) -> None:
        if overwrite or not self.root().exists():
            await self._downloader.download(self._source, self.root(), overwrite=overwrite)
        await self.resolve()

    def status(self) -> ModelResourceStatus:
        root = self.root()
        available = all((root / name).is_file() for name in self._artifact.required_files)
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml-int8",
                    format="coreml",
                    runtime="coreml",
                    available=available,
                    size_bytes=directory_size(root) if available else None,
                ),
            ),
        )

    async def delete(self) -> None:
        root = self.root().expanduser().resolve(strict=False)
        model_root = self._artifact.storage_path(self._config.model_home).parents[2]
        if root != model_root and not root.is_relative_to(model_root):
            raise ResourceIntegrityError(
                "Refusing to delete DeepFilterNet3 resources outside model root"
            )
        await asyncio.to_thread(shutil.rmtree, root, True)


class DeepFilterNet3ResourceProvider:
    def __init__(
        self, source: HuggingFaceSource, manifest: ModelManifest, artifact: ModelArtifact
    ) -> None:
        self._source = source
        self._manifest, self._artifact = manifest, artifact

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> DeepFilterNet3ResourceResolver:
        return DeepFilterNet3ResourceResolver(
            self._source,
            DeepFilterNet3InstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
            manifest=self._manifest,
            artifact=self._artifact,
        )

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.download(overwrite=overwrite)
        return resolver.status()

    async def convert(
        self,
        variant: str,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        del variant, options, overwrite
        raise UnsupportedRuntimeError(
            "DeepFilterNet3 ships a prebuilt Core ML artifact and cannot "
            f"convert to {target_format}"
        )

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime not in {None, "coreml"}:
            raise UnsupportedRuntimeError(f"Unsupported runtime: {runtime}")
        resolver = self._resolver(variant, options)
        await resolver.delete()
        return resolver.status()
