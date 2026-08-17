"""Variant-aware resource management for Gemma 4 MLX."""

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
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    GEMMA_4_MLX_REQUIRED_FILES,
    GEMMA_4_MLX_VARIANTS,
    GEMMA_4_MODEL_ID,
    GEMMA_4_TOKENIZER_REQUIRED_FILES,
    GEMMA_4_TOKENIZER_REVISION,
    Gemma4MlxInstanceConfig,
)


class Gemma4MlxResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Gemma4MlxInstanceConfig,
        *,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return (
            self._config.source_path
            or self._config.model_home
            / "google"
            / "gemma-4"
            / self._config.variant
            / self._source.revision
            / "mlx"
            / "model"
        )

    @property
    def tokenizer_path(self) -> Path:
        return (
            self._config.tokenizer_path
            or self._config.model_home
            / "google"
            / "gemma-4"
            / "shared"
            / "tokenizer"
            / GEMMA_4_TOKENIZER_REVISION
        )

    async def resolve_source(self) -> ResolvedResource:
        self._validate(self.path, self._config.variant)
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in GEMMA_4_TOKENIZER_REQUIRED_FILES
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Gemma 4 shared tokenizer is incomplete", details={"missing": missing}
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
            (self.path / name).is_file()
            for name in GEMMA_4_MLX_REQUIRED_FILES[self._config.variant]
        )
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file() for name in GEMMA_4_TOKENIZER_REQUIRED_FILES
        )
        return ModelResourceStatus(
            model_id=GEMMA_4_MODEL_ID,
            revision=self._source.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="mlx-optiq-4bit",
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="tokenizer",
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=directory_size(self.tokenizer_path) if tokenizer_available else None,
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"Gemma 4 does not support runtime {runtime}")
        target = self.path.expanduser().resolve(strict=False)
        root = (
            (self._config.model_home / "google" / "gemma-4" / self._config.variant)
            .expanduser()
            .resolve(strict=False)
        )
        if not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Gemma 4 resources outside the variant directory"
            )
        await asyncio.to_thread(shutil.rmtree, target if runtime else root, True)

    @staticmethod
    def _validate(path: Path, variant: str) -> None:
        missing = [
            name for name in GEMMA_4_MLX_REQUIRED_FILES[variant] if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Gemma 4 MLX snapshot is incomplete", details={"missing": missing}
            )


class Gemma4MlxResourceProvider:
    def __init__(
        self, sources: Mapping[str, HuggingFaceSource], tokenizer_source: HuggingFaceSource
    ) -> None:
        self._sources = dict(sources)
        self._tokenizer_source = tokenizer_source

    def _resolver(
        self, variant: str, options: Mapping[str, object] | None
    ) -> Gemma4MlxResourceResolver:
        if variant not in GEMMA_4_MLX_VARIANTS:
            raise ResourceNotFoundError(f"Gemma 4 variant is not registered: {variant}")
        normalized = dict(options or {})
        if normalized.get("variant") not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Gemma 4 variant values were provided")
        return Gemma4MlxResourceResolver(
            self._sources[variant],
            Gemma4MlxInstanceConfig.model_validate(normalized | {"variant": variant}),
            tokenizer_source=self._tokenizer_source,
        )

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
        raise UnsupportedRuntimeError(f"Gemma 4 conversion to {target_format} is not implemented")

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
