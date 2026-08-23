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
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import Audio8TtsMlxInstanceConfig


class Audio8TtsMlxResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8TtsMlxInstanceConfig,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    @property
    def path(self) -> Path:
        return self._config.source_path or self._artifact.resolve(self._config.model_home)

    @property
    def tokenizer_path(self) -> Path:
        return self._config.tokenizer_path or self._tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_source(self) -> ResolvedResource:
        self._validate(self.path)
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in self._tokenizer_artifact.required_files
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS shared tokenizer is incomplete", details={"missing": missing}
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
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._source.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="mlx-bf16",
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    required_shares=self._artifact.required_shares,
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
        return self._artifact.storage_path(self._config.model_home).parents[1]

    def _validate(self, path: Path) -> None:
        missing = [name for name in self._artifact.required_files if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS MLX BF16 snapshot is incomplete", details={"missing": missing}
            )


class Audio8TtsMlxResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource | None = None,
        *,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        tokenizer_artifact: ModelArtifact,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source
        self._manifest = manifest
        self._artifact = artifact
        self._tokenizer_artifact = tokenizer_artifact

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
    ) -> Audio8TtsMlxResourceResolver:
        if variant != self._artifact.variant:
            raise ResourceNotFoundError(f"Audio8-TTS MLX variant is not registered: {variant}")
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Audio8-TTS MLX variant values were provided")
        return Audio8TtsMlxResourceResolver(
            self._source,
            Audio8TtsMlxInstanceConfig.model_validate(normalized | {"variant": variant}),
            manifest=self._manifest,
            artifact=self._artifact,
            tokenizer_artifact=self._tokenizer_artifact,
            tokenizer_source=self._tokenizer_source,
        )
