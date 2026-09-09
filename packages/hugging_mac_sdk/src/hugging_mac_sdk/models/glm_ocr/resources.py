"""Resource lifecycle for the GLM-OCR eight-bit checkpoint."""

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
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import GlmOcrMlxInstanceConfig


class GlmOcrResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: GlmOcrMlxInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._manifest, self._artifact = manifest, artifact
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.artifact_path or self._artifact.resolve(self._config.model_home)

    async def resolve(self) -> ResolvedResource:
        missing = [
            name for name in self._artifact.required_files if not (self.path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "GLM-OCR checkpoint is incomplete", details={"missing": missing}
            )
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def download(self, *, overwrite: bool = False) -> None:
        complete = self.path.is_dir() and all(
            (self.path / name).is_file() for name in self._artifact.required_files
        )
        if overwrite or not complete:
            await self._downloader.download(
                self._source, self.path, overwrite=self.path.exists(), token=self._config.hf_token
            )
        await self.resolve()

    def status(self) -> ModelResourceStatus:
        available = self.path.is_dir() and all(
            (self.path / name).is_file() for name in self._artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id=self._artifact.artifact_id,
                    format=self._artifact.format.value,
                    runtime="mlx",
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
            ),
        )

    async def delete(self) -> None:
        target = self.path.expanduser().resolve(strict=False)
        expected = (
            self._artifact.resolve(self._config.model_home).expanduser().resolve(strict=False)
        )
        if target != expected:
            raise ResourceIntegrityError("Refusing to delete an externally supplied GLM-OCR path")
        await asyncio.to_thread(shutil.rmtree, target, True)


class GlmOcrResourceProvider:
    def __init__(
        self, source: HuggingFaceSource, manifest: ModelManifest, artifact: ModelArtifact
    ) -> None:
        self._source, self._manifest, self._artifact = source, manifest, artifact

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> GlmOcrResourceResolver:
        config = GlmOcrMlxInstanceConfig.model_validate(
            dict(options or {}) | {"variant": variant, "runtime": "mlx"}
        )
        return GlmOcrResourceResolver(
            self._source, config, manifest=self._manifest, artifact=self._artifact
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
        raise UnsupportedRuntimeError(f"GLM-OCR cannot convert to {target_format}")

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"GLM-OCR does not support runtime {runtime}")
        resolver = self._resolver(variant, options)
        await resolver.delete()
        return resolver.status()
