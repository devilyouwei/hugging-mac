"""Audio8-TTS resource resolution, download, status, and deletion."""

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
    AUDIO8_TTS_CODEC_SHA256,
    AUDIO8_TTS_MODEL_ID,
    AUDIO8_TTS_REQUIRED_FILES,
    AUDIO8_TTS_REVISION,
    AUDIO8_TTS_TOKENIZER_REQUIRED_FILES,
    AUDIO8_TTS_VARIANT,
    AUDIO8_TTS_WEIGHT_SHA256,
    Audio8TtsInstanceConfig,
)
from .mlx_resources import Audio8TtsMlxResourceProvider


class Audio8TtsResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8TtsInstanceConfig,
        *,
        tokenizer_source: HuggingFaceSource | None = None,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=3600)

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        digest = await asyncio.to_thread(self._validate_snapshot, path)
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=directory_size(path),
        )

    @property
    def tokenizer_path(self) -> Path:
        return (
            self._config.tokenizer_path
            or self._config.model_home
            / "audio8"
            / "audio8-tts-preview"
            / "shared"
            / "tokenizer"
            / "afa762738efa064eb11720c934271b739b02cf97"
        )

    async def resolve_tokenizer(self) -> ResolvedResource:
        missing = [
            name
            for name in AUDIO8_TTS_TOKENIZER_REQUIRED_FILES
            if not (self.tokenizer_path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS shared tokenizer is incomplete",
                details={"model_id": AUDIO8_TTS_MODEL_ID, "missing": missing},
            )
        return ResolvedResource(
            path=self.tokenizer_path,
            source=self._tokenizer_source,
            size_bytes=directory_size(self.tokenizer_path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if overwrite or not path.exists():
            await self._downloader.download(
                self._source,
                path,
                overwrite=overwrite,
                token=self._config.hf_token,
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

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime not in {None, "pytorch"}:
            raise UnsupportedRuntimeError(
                f"Audio8-TTS does not support runtime {runtime}",
                details={"model_id": AUDIO8_TTS_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        target = root if runtime is None else (
            self._config.source_path or self._default_source_path()
        )
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Audio8-TTS resources outside the model directory",
                details={"model_id": AUDIO8_TTS_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        path = self._config.source_path or self._default_source_path()
        available = self._snapshot_files_exist(path)
        tokenizer_available = all(
            (self.tokenizer_path / name).is_file()
            for name in AUDIO8_TTS_TOKENIZER_REQUIRED_FILES
        )
        return ModelResourceStatus(
            model_id=AUDIO8_TTS_MODEL_ID,
            revision=AUDIO8_TTS_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.SAFETENSORS.value,
                    runtime="pytorch",
                    available=available,
                    size_bytes=directory_size(path) if available else None,
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

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "audio8"
            / "audio8-tts-preview"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).is_file() for name in AUDIO8_TTS_REQUIRED_FILES
        )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "Audio8-TTS source model is not downloaded",
                details={"model_id": AUDIO8_TTS_MODEL_ID, "artifact_id": "source"},
            )
        missing = [
            name for name in AUDIO8_TTS_REQUIRED_FILES if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS snapshot is incomplete",
                details={"model_id": AUDIO8_TTS_MODEL_ID, "missing": missing},
            )
        digest = file_sha256(path / "model.safetensors")
        if digest != AUDIO8_TTS_WEIGHT_SHA256:
            raise ResourceIntegrityError(
                "Audio8-TTS weight SHA-256 mismatch",
                details={"expected": AUDIO8_TTS_WEIGHT_SHA256, "actual": digest},
            )
        codec_digest = file_sha256(path / "codec.pth")
        if codec_digest != AUDIO8_TTS_CODEC_SHA256:
            raise ResourceIntegrityError(
                "Audio8-TTS codec SHA-256 mismatch",
                details={"expected": AUDIO8_TTS_CODEC_SHA256, "actual": codec_digest},
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "arktts":
            raise ResourceIntegrityError(
                "Audio8-TTS config has an unexpected model_type",
                details={"model_type": config.get("model_type")},
            )
        return digest

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Audio8TtsResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource | None = None,
    ) -> None:
        self._source = source
        self._tokenizer_source = tokenizer_source or source

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
        del options, overwrite
        raise UnsupportedRuntimeError(
            f"Audio8-TTS conversion to {target_format} is not implemented",
            details={"model_id": AUDIO8_TTS_MODEL_ID, "variant": variant},
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
    ) -> Audio8TtsResourceResolver:
        if variant != AUDIO8_TTS_VARIANT:
            raise ResourceNotFoundError(
                f"Audio8-TTS variant is not registered: {variant}",
                details={"supported_variants": [AUDIO8_TTS_VARIANT]},
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting Audio8-TTS variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = Audio8TtsInstanceConfig.model_validate(
            normalized | {"variant": variant}
        )
        return Audio8TtsResourceResolver(
            self._source, config, tokenizer_source=self._tokenizer_source
        )


class Audio8TtsCombinedResourceProvider:
    """Expose all downloadable runtime artifacts through one canonical model."""

    def __init__(
        self,
        pytorch_source: HuggingFaceSource,
        mlx_source: HuggingFaceSource,
        tokenizer_source: HuggingFaceSource,
    ) -> None:
        self._pytorch = Audio8TtsResourceProvider(pytorch_source, tokenizer_source)
        self._mlx = Audio8TtsMlxResourceProvider(mlx_source, tokenizer_source)

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        pytorch = await self._pytorch.status(variant, options)
        mlx = await self._mlx.status(variant, options)
        # Reconstruct the status so ModelResourceStatus recomputes its derived
        # runtime aggregates for both backends. model_copy(update=...) skips
        # validation and would leave only the PyTorch runtime visible.
        return ModelResourceStatus(
            model_id=pytorch.model_id,
            revision=pytorch.revision,
            variant=pytorch.variant,
            artifacts=pytorch.artifacts + mlx.artifacts,
        )

    async def download_source(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        await self._pytorch.download_source(variant, options, overwrite=overwrite)
        await self._mlx.download_source(variant, options, overwrite=overwrite)
        return await self.status(variant, options)

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
            f"Audio8-TTS conversion to {target_format} is not implemented"
        )

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        if runtime in {None, "pytorch"}:
            await self._pytorch.delete(variant, options, runtime=runtime)
        if runtime in {None, "mlx"}:
            await self._mlx.delete(variant, options, runtime=runtime)
        if runtime not in {None, "pytorch", "mlx"}:
            raise UnsupportedRuntimeError(f"Audio8-TTS does not support runtime {runtime}")
        return await self.status(variant, options)
