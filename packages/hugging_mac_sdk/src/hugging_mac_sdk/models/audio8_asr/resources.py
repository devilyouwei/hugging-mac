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
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
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


class Audio8AsrResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8AsrInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader()

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

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "pytorch-mps"}:
            raise UnsupportedRuntimeError(
                f"Audio8-ASR does not support runtime {runtime}",
                details={"model_id": AUDIO8_ASR_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        target = (
            root if runtime is None else (self._config.source_path or self._default_source_path())
        )
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
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

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
        del options, overwrite
        raise UnsupportedRuntimeError(
            f"Audio8-ASR conversion to {target_format} is not implemented",
            details={
                "model_id": AUDIO8_ASR_MODEL_ID,
                "variant": variant,
                "target_format": target_format,
            },
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
