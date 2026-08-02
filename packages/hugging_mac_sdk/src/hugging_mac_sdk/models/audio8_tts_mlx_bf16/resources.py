"""Resource management for Audio8-TTS MLX BF16."""

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
    AUDIO8_TTS_MLX_BF16_MODEL_ID,
    AUDIO8_TTS_MLX_BF16_REQUIRED_FILES,
    AUDIO8_TTS_MLX_BF16_REVISION,
    AUDIO8_TTS_MLX_BF16_VARIANT,
    Audio8TtsMlxBf16InstanceConfig,
)


class Audio8TtsMlxBf16ResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8TtsMlxBf16InstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.source_path or self._model_root() / "mlx" / "model"

    async def resolve_source(self) -> ResolvedResource:
        self._validate(self.path)
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
            (self.path / name).is_file() for name in AUDIO8_TTS_MLX_BF16_REQUIRED_FILES
        )
        return ModelResourceStatus(
            model_id=AUDIO8_TTS_MLX_BF16_MODEL_ID,
            revision=AUDIO8_TTS_MLX_BF16_REVISION,
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
            raise UnsupportedRuntimeError(f"Audio8-TTS MLX does not support runtime {runtime}")
        root = self._model_root().expanduser().resolve(strict=False)
        target = (root if runtime is None else self.path).expanduser().resolve(strict=False)
        if target != root and not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Audio8-TTS MLX resources outside the model directory"
            )
        await asyncio.to_thread(shutil.rmtree, target, True)

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "mlx-community"
            / "audio8-tts-preview-0.6b-bf16"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _validate(path: Path) -> None:
        missing = [
            name for name in AUDIO8_TTS_MLX_BF16_REQUIRED_FILES if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS MLX BF16 snapshot is incomplete", details={"missing": missing}
            )


class Audio8TtsMlxBf16ResourceProvider:
    def __init__(self, source: HuggingFaceSource) -> None:
        self._source = source

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
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
        raise UnsupportedRuntimeError(
            f"Audio8-TTS MLX conversion to {target_format} is not implemented"
        )

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
    ) -> Audio8TtsMlxBf16ResourceResolver:
        if variant != AUDIO8_TTS_MLX_BF16_VARIANT:
            raise ResourceNotFoundError(f"Audio8-TTS MLX variant is not registered: {variant}")
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Audio8-TTS MLX variant values were provided")
        return Audio8TtsMlxBf16ResourceResolver(
            self._source,
            Audio8TtsMlxBf16InstanceConfig.model_validate(normalized | {"variant": variant}),
        )
