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
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    QWEN3_5_MLX_MODEL_ID,
    QWEN3_5_MLX_REQUIRED_FILES,
    QWEN3_5_MLX_VARIANTS,
    Qwen35MlxInstanceConfig,
)


class Qwen35MlxResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Qwen35MlxInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source, self._config = source, config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return (
            self._config.source_path
            or self._config.model_home
            / "mlx-community"
            / "qwen3.5-mlx"
            / self._config.variant
            / self._source.revision
            / "mlx"
            / "model"
        )

    async def resolve_source(self) -> ResolvedResource:
        self._validate(self.path, self._config.variant)
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if self.path.exists() and not overwrite:
            return await self.resolve_source()
        await self._downloader.download(
            self._source, self.path, overwrite=overwrite, token=self._config.hf_token
        )
        return await self.resolve_source()

    def status(self) -> ModelResourceStatus:
        available = self.path.is_dir() and all(
            (self.path / name).is_file()
            for name in QWEN3_5_MLX_REQUIRED_FILES[self._config.variant]
        )
        return ModelResourceStatus(
            model_id=QWEN3_5_MLX_MODEL_ID,
            revision=self._source.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="model",
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    available=available,
                    size_bytes=directory_size(self.path) if available else None,
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"Qwen3.5 does not support runtime {runtime}")
        target = self.path.expanduser().resolve(strict=False)
        root = (
            (self._config.model_home / "mlx-community" / "qwen3.5-mlx" / self._config.variant)
            .expanduser()
            .resolve(strict=False)
        )
        if not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3.5 resources outside the variant directory"
            )
        await asyncio.to_thread(shutil.rmtree, target if runtime else root, True)

    @staticmethod
    def _validate(path: Path, variant: str) -> None:
        missing = [
            name for name in QWEN3_5_MLX_REQUIRED_FILES[variant] if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Qwen3.5 MLX snapshot is incomplete", details={"missing": missing}
            )


class Qwen35MlxResourceProvider:
    def __init__(self, sources: Mapping[str, HuggingFaceSource]) -> None:
        self._sources = dict(sources)

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
        if variant not in QWEN3_5_MLX_VARIANTS:
            raise ResourceNotFoundError(f"Qwen3.5 variant is not registered: {variant}")
        normalized = dict(options or {})
        if normalized.get("variant") not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Qwen3.5 variant values were provided")
        return Qwen35MlxResourceResolver(
            self._sources[variant],
            Qwen35MlxInstanceConfig.model_validate(normalized | {"variant": variant}),
        )
