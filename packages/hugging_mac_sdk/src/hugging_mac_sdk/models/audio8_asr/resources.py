"""Audio8-ASR resource resolution, download, status, and deletion."""

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
from hugging_mac_sdk.resources.hashing import directory_size, file_sha256
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    AUDIO8_ASR_MODEL_ID,
    AUDIO8_ASR_REQUIRED_FILES,
    AUDIO8_ASR_REVISION,
    AUDIO8_ASR_VARIANT,
    AUDIO8_ASR_WEIGHT_SHA256,
    Audio8AsrInstanceConfig,
)
from .converter import Audio8AsrConverter


class Audio8AsrResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8AsrInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: Audio8AsrConverter | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter or Audio8AsrConverter()

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        digest = await asyncio.to_thread(self._validate_snapshot, path)
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=directory_size(path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if path.exists() and not overwrite:
            return await self.resolve_source()
        await self._downloader.download(
            self._source,
            path,
            overwrite=overwrite,
            token=self._config.hf_token,
        )
        return await self.resolve_source()

    async def resolve_coreml(self) -> ResolvedResource:
        path = self._config.artifact_path or self._default_coreml_path()
        await asyncio.to_thread(self._validate_coreml_artifact, path)
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=None,
            size_bytes=directory_size(path),
        )

    async def convert_coreml(self, *, overwrite: bool = False) -> None:
        path = self._config.artifact_path or self._default_coreml_path()
        if path.exists() and not overwrite:
            await self.resolve_coreml()
            return
        source = await self.resolve_source()
        await self._converter.convert(
            ConversionRequest(
                model_id=AUDIO8_ASR_MODEL_ID,
                model_revision=AUDIO8_ASR_REVISION,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.SAFETENSORS,
                target_format=ArtifactFormat.COREML,
                output_path=path,
                overwrite=overwrite,
            )
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "pytorch-mps", "coreml"}:
            raise UnsupportedRuntimeError(
                f"Audio8-ASR does not support runtime {runtime}",
                details={"model_id": AUDIO8_ASR_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        if runtime is None:
            target = root
        elif runtime == "pytorch-mps":
            target = self._config.source_path or self._default_source_path()
        else:
            target = self._config.artifact_path or self._default_coreml_path()
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Audio8-ASR resources outside the model directory",
                details={"model_id": AUDIO8_ASR_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        path = self._config.source_path or self._default_source_path()
        available = self._snapshot_files_exist(path)
        coreml_path = self._config.artifact_path or self._default_coreml_path()
        coreml_available = self._coreml_files_exist(coreml_path)
        return ModelResourceStatus(
            model_id=AUDIO8_ASR_MODEL_ID,
            revision=AUDIO8_ASR_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.SAFETENSORS.value,
                    runtime="pytorch-mps",
                    available=available,
                    size_bytes=directory_size(path) if available else None,
                ),
                ModelArtifactStatus(
                    artifact_id="coreml",
                    format=ArtifactFormat.COREML.value,
                    runtime="coreml",
                    provisioning="convert",
                    available=coreml_available,
                    size_bytes=directory_size(coreml_path) if coreml_available else None,
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

    def _default_coreml_path(self) -> Path:
        return self._model_root() / "coreml" / "audio8-asr-coreml"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "audio8"
            / "audio8-asr-0.1b"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in AUDIO8_ASR_REQUIRED_FILES)

    @staticmethod
    def _coreml_files_exist(path: Path) -> bool:
        required = (
            "audio_tower.mlpackage",
            "language_model.safetensors",
            "projector.safetensors",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "conversion.json",
        )
        return path.is_dir() and all((path / name).exists() for name in required)

    def _validate_coreml_artifact(self, path: Path) -> None:
        if not self._coreml_files_exist(path):
            raise ResourceNotFoundError(
                "Audio8-ASR Core ML artifact has not been converted",
                details={
                    "model_id": AUDIO8_ASR_MODEL_ID,
                    "variant": self._config.variant,
                    "artifact_id": "coreml",
                },
            )
        metadata = json.loads((path / "conversion.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "audio8-asr-coreml-hybrid":
            raise ResourceIntegrityError(
                "Audio8-ASR Core ML artifact metadata is invalid",
                details={"format": metadata.get("format")},
            )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "Audio8-ASR source model is not downloaded",
                details={
                    "model_id": AUDIO8_ASR_MODEL_ID,
                    "variant": self._config.variant,
                    "artifact_id": "source",
                },
            )
        missing = [name for name in AUDIO8_ASR_REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-ASR snapshot is incomplete",
                details={"model_id": AUDIO8_ASR_MODEL_ID, "missing": missing},
            )
        digest = file_sha256(path / "model.safetensors")
        if digest != AUDIO8_ASR_WEIGHT_SHA256:
            raise ResourceIntegrityError(
                "Audio8-ASR weight SHA-256 mismatch",
                details={"expected": AUDIO8_ASR_WEIGHT_SHA256, "actual": digest},
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "arkasr":
            raise ResourceIntegrityError(
                "Audio8-ASR config has an unexpected model_type",
                details={"model_type": config.get("model_type")},
            )
        return digest

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Audio8AsrResourceProvider:
    def __init__(self, source: HuggingFaceSource) -> None:
        self._source = source

    async def status(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
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
        if target_format is not ArtifactFormat.COREML:
            raise UnsupportedRuntimeError(
                f"Audio8-ASR conversion to {target_format} is not implemented",
                details={
                    "model_id": AUDIO8_ASR_MODEL_ID,
                    "variant": variant,
                    "target_format": target_format,
                },
            )
        resolver = self._resolver(variant, options)
        await resolver.convert_coreml(overwrite=overwrite)
        return resolver.status()

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
        self,
        variant: str,
        options: Mapping[str, object] | None,
    ) -> Audio8AsrResourceResolver:
        if variant != AUDIO8_ASR_VARIANT:
            raise ResourceNotFoundError(
                f"Audio8-ASR variant is not registered: {variant}",
                details={
                    "model_id": AUDIO8_ASR_MODEL_ID,
                    "variant": variant,
                    "supported_variants": [AUDIO8_ASR_VARIANT],
                },
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting Audio8-ASR variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = Audio8AsrInstanceConfig.model_validate(normalized | {"variant": variant})
        return Audio8AsrResourceResolver(self._source, config)
