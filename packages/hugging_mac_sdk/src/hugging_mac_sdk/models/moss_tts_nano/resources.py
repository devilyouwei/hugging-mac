"""Managed resources for MOSS-TTS-Nano."""

from __future__ import annotations

import asyncio
import json
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

from .config import MossTtsNanoInstanceConfig


class MossTtsNanoResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        audio_tokenizer_source: HuggingFaceSource,
        config: MossTtsNanoInstanceConfig,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        audio_tokenizer_artifact: ModelArtifact,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._audio_tokenizer_source = audio_tokenizer_source
        self._config = config
        self._manifest = manifest
        self._artifact = artifact
        self._audio_tokenizer_artifact = audio_tokenizer_artifact
        self._downloader = downloader or ResourceDownloader(timeout=1800)

    @property
    def path(self) -> Path:
        return self._config.source_path or self._artifact.resolve(self._config.model_home)

    @property
    def audio_tokenizer_path(self) -> Path:
        return self._config.audio_tokenizer_path or self._audio_tokenizer_artifact.resolve(
            self._config.model_home
        )

    async def resolve_source(self) -> ResolvedResource:
        await asyncio.to_thread(self._validate, self.path, self._artifact, "moss_tts_nano", "model")
        return ResolvedResource(
            path=self.path, source=self._source, size_bytes=directory_size(self.path)
        )

    async def resolve_audio_tokenizer(self) -> ResolvedResource:
        await asyncio.to_thread(
            self._validate,
            self.audio_tokenizer_path,
            self._audio_tokenizer_artifact,
            "moss-audio-tokenizer",
            "audio tokenizer",
        )
        return ResolvedResource(
            path=self.audio_tokenizer_path,
            source=self._audio_tokenizer_source,
            size_bytes=directory_size(self.audio_tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if overwrite or not self.path.exists():
            await self._downloader.download(
                self._source, self.path, overwrite=overwrite, token=self._config.hf_token
            )
        if overwrite or not self.audio_tokenizer_path.exists():
            await self._downloader.download(
                self._audio_tokenizer_source,
                self.audio_tokenizer_path,
                overwrite=overwrite,
                token=self._config.hf_token,
            )
        await self.resolve_audio_tokenizer()
        return await self.resolve_source()

    def status(self) -> ModelResourceStatus:
        model_available = self._files_exist(self.path, self._artifact)
        tokenizer_available = self._files_exist(
            self.audio_tokenizer_path, self._audio_tokenizer_artifact
        )
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id=self._artifact.artifact_id,
                    format=ArtifactFormat.MLX.value,
                    runtime="mlx",
                    required_shares=self._artifact.required_shares,
                    available=model_available,
                    size_bytes=directory_size(self.path) if model_available else None,
                ),
                ModelArtifactStatus(
                    artifact_id=self._audio_tokenizer_artifact.artifact_id,
                    format=ArtifactFormat.TOKENIZER.value,
                    runtime=None,
                    shared=True,
                    available=tokenizer_available,
                    size_bytes=(
                        directory_size(self.audio_tokenizer_path) if tokenizer_available else None
                    ),
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "mlx"}:
            raise UnsupportedRuntimeError(f"MOSS-TTS-Nano does not support runtime {runtime}")
        root = self._model_root().expanduser().resolve(strict=False)
        target = (root if runtime is None else self.path).expanduser().resolve(strict=False)
        if target != root and not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete MOSS-TTS-Nano resources outside the model directory"
            )
        await asyncio.to_thread(shutil.rmtree, target, True)

    def _model_root(self) -> Path:
        return self._artifact.storage_path(self._config.model_home).parents[1]

    @staticmethod
    def _files_exist(path: Path, artifact: ModelArtifact) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in artifact.required_files)

    @classmethod
    def _validate(
        cls,
        path: Path,
        artifact: ModelArtifact,
        expected_model_type: str,
        label: str,
    ) -> None:
        missing = [name for name in artifact.required_files if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                f"MOSS-TTS-Nano {label} is incomplete", details={"missing": missing}
            )
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResourceIntegrityError(f"MOSS-TTS-Nano {label} config is invalid") from error
        if config.get("model_type") != expected_model_type:
            raise ResourceIntegrityError(
                f"MOSS-TTS-Nano {label} has an unsupported model_type",
                details={"model_type": config.get("model_type")},
            )


class MossTtsNanoResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        audio_tokenizer_source: HuggingFaceSource,
        manifest: ModelManifest,
        artifact: ModelArtifact,
        audio_tokenizer_artifact: ModelArtifact,
    ) -> None:
        self._source = source
        self._audio_tokenizer_source = audio_tokenizer_source
        self._manifest = manifest
        self._artifact = artifact
        self._audio_tokenizer_artifact = audio_tokenizer_artifact

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
            f"MOSS-TTS-Nano conversion to {target_format} is not implemented"
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
    ) -> MossTtsNanoResourceResolver:
        if variant not in {item.name for item in self._manifest.variants}:
            raise ResourceNotFoundError(f"MOSS-TTS-Nano variant is not registered: {variant}")
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError("Conflicting MOSS-TTS-Nano variant values were provided")
        config = MossTtsNanoInstanceConfig.model_validate(normalized | {"variant": variant})
        return MossTtsNanoResourceResolver(
            self._source,
            self._audio_tokenizer_source,
            config,
            self._manifest,
            self._artifact,
            self._audio_tokenizer_artifact,
        )
