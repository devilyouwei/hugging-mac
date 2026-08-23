"""Resource lifecycle for prebuilt Nemotron 3.5 ASR Core ML bundles."""

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

from .config import NemotronCoreMlInstanceConfig


def _variant_source_path(variant: str) -> str:
    script, tier = variant.split("-", maxsplit=1)
    return f"{script}/{tier}"


class NemotronCoreMlResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: NemotronCoreMlInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._manifest = manifest
        self._artifact = artifact
        self._downloader = downloader or ResourceDownloader()

    def root(self) -> Path:
        if self._config.artifact_path is not None:
            return self._config.artifact_path
        return self._artifact.resolve(self._config.model_home)

    def bundle_root(self) -> Path:
        root = self.root()
        nested = root / _variant_source_path(self._config.variant)
        return nested if nested.exists() else root

    async def resolve(self) -> ResolvedResource:
        root = self.bundle_root()
        missing = [name for name in self._artifact.required_files if not (root / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Nemotron Core ML artifact is not downloaded", details={"missing": missing}
            )
        return ResolvedResource(
            path=root,
            source=self._source,
            digest=directory_sha256(root),
            size_bytes=directory_size(root),
        )

    def status(self) -> ModelResourceStatus:
        root = self.bundle_root()
        available = all((root / name).is_file() for name in self._artifact.required_files)
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="coreml-mixed",
                    format="coreml",
                    runtime="coreml",
                    available=available,
                    size_bytes=directory_size(root) if available else None,
                ),
            ),
        )

    async def download(self, *, overwrite: bool = False) -> None:
        if overwrite or not self.root().exists():
            await self._downloader.download(self._source, self.root(), overwrite=overwrite)
        await self.resolve()

    async def delete(self) -> None:
        root = self.root().expanduser().resolve(strict=False)
        model_root = self._artifact.storage_path(self._config.model_home).parents[2]
        if root != model_root and not root.is_relative_to(model_root):
            raise ResourceIntegrityError("Refusing to delete Nemotron resources outside model root")
        await asyncio.to_thread(shutil.rmtree, root, True)


class NemotronCoreMlResourceProvider:
    def __init__(
        self,
        sources: Mapping[str, HuggingFaceSource],
        artifacts: Mapping[str, ModelArtifact],
        manifest: ModelManifest,
    ) -> None:
        self._sources = dict(sources)
        self._artifacts = dict(artifacts)
        self._manifest = manifest

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> NemotronCoreMlResourceResolver:
        return NemotronCoreMlResourceResolver(
            self._sources[variant],
            NemotronCoreMlInstanceConfig.model_validate(dict(options or {}) | {"variant": variant}),
            manifest=self._manifest,
            artifact=self._artifacts[variant],
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
            f"Nemotron ships prebuilt Core ML graphs and cannot convert to {target_format}"
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
