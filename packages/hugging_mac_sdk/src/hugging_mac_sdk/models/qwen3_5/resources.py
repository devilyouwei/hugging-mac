"""Variant-aware resource management for Qwen3.5 MLX."""

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

from .config import Qwen35MlxInstanceConfig


class Qwen35MlxResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Qwen35MlxInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._manifest, self._artifact = manifest, artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.source_path or self._artifact.resolve(self._config.model_home)

    async def resolve_source(self) -> ResolvedResource:
        self._validate(self.path, self._config.variant)
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in self._tokenizer_artifact.required_files
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3.5 shared tokenizer is incomplete", details={"missing": missing}
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            size_bytes=directory_size(self.tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if overwrite or not self.path.exists():
            await self._downloader.download(
                self._source, self.path, overwrite=overwrite, token=self._config.hf_token
            )
        if overwrite or not self.tokenizer_path.exists():
            await self._downloader.download(
                self._tokenizer_source,
                self.tokenizer_path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        await self.resolve_tokenizer()
        return await self.resolve_source()

    def status(self) -> ModelResourceStatus:
        available = self.path.is_dir() and all(
            (self.path / name).is_file() for name in self._artifact.required_files
        )
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file()
            for name in self._tokenizer_artifact.required_files
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._source.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="mlx-optiq-4bit",
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    required_shares=self._artifact.required_shares,
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizer",
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=(
                        directory_size(self.tokenizer_path) if tokenizer_available else None
                    ),
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"Qwen3.5 does not support runtime {runtime}")
        target = self.path.expanduser().resolve(strict=False)
        root = self._artifact.storage_path(self._config.model_home).parents[1]
        if not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3.5 resources outside the variant directory"
            )
        await asyncio.to_thread(shutil.rmtree, target if runtime else root, True)

    def _validate(self, path: Path, variant: str) -> None:
        del variant
        missing = [name for name in self._artifact.required_files if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3.5 MLX snapshot is incomplete", details={"missing": missing}
            )


class Qwen35MlxResourceProvider:
    def __init__(
        self,
        sources: Mapping[str, HuggingFaceSource],
        tokenizer_source: HuggingFaceSource,
        manifest: ModelManifest,
        artifacts: Mapping[str, ModelArtifact],
        tokenizer_artifact: ModelArtifact,
    ) -> None:
        self._sources = dict(sources)
        self._tokenizer_source = tokenizer_source
        self._manifest, self._artifacts = manifest, dict(artifacts)
        self._tokenizer_artifact = tokenizer_artifact

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.download_source(overwrite=overwrite)
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
        raise UnsupportedRuntimeError(f"Qwen3.5 conversion to {target_format} is not implemented")

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options)
        await resolver.delete(runtime=runtime)
        return resolver.status()

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> Qwen35MlxResourceResolver:
        if variant not in self._artifacts:
            raise ResourceNotFoundError(f"Qwen3.5 variant is not registered: {variant}")
        normalized = dict(options or {})
        if normalized.get("variant") not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Qwen3.5 variant values were provided")
        return Qwen35MlxResourceResolver(
            self._sources[variant],
            Qwen35MlxInstanceConfig.model_validate(normalized | {"variant": variant}),
            manifest=self._manifest,
            artifact=self._artifacts[variant],
            tokenizer_artifact=self._tokenizer_artifact,
            tokenizer_source=self._tokenizer_source,
        )
