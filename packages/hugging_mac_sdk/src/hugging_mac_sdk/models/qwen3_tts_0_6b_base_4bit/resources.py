"""Resource management for Qwen3-TTS 0.6B Base MLX 4-bit."""

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
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import (
    QWEN3_TTS_MODEL_ID,
    QWEN3_TTS_REQUIRED_FILES,
    QWEN3_TTS_REVISION,
    QWEN3_TTS_VARIANT,
    Qwen3TtsInstanceConfig,
)


class Qwen3TtsResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Qwen3TtsInstanceConfig,
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
        await asyncio.to_thread(self._validate, self.path)
        return ResolvedResource(
            path=self.path,
            source=self._source,
            size_bytes=directory_size(self.path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if self.path.exists() and not overwrite:
            return await self.resolve_source()
        await self._downloader.download(
            self._source,
            self.path,
            overwrite=overwrite,
            token=self._config.hf_token,
        )
        return await self.resolve_source()

    def status(self) -> ModelResourceStatus:
        available = self._files_exist(self.path)
        return ModelResourceStatus(
            model_id=QWEN3_TTS_MODEL_ID,
            revision=QWEN3_TTS_REVISION,
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
            raise UnsupportedRuntimeError(f"Qwen3-TTS does not support runtime {runtime}")
        root = self._model_root().expanduser().resolve(strict=False)
        target = (root if runtime is None else self.path).expanduser().resolve(strict=False)
        if target != root and not target.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Qwen3-TTS resources outside the model directory"
            )
        await asyncio.to_thread(shutil.rmtree, target, True)

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "mlx-community"
            / "qwen3-tts-12hz-0.6b-base-4bit"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _files_exist(path: Path) -> bool:
        return (
            path.is_dir()
            and all((path / name).is_file() for name in QWEN3_TTS_REQUIRED_FILES)
            and any(path.glob("*.safetensors"))
            and any((path / "speech_tokenizer").glob("*.safetensors"))
        )

    @classmethod
    def _validate(cls, path: Path) -> None:
        missing = [name for name in QWEN3_TTS_REQUIRED_FILES if not (path / name).is_file()]
        if not any(path.glob("*.safetensors")):
            missing.append("*.safetensors")
        if not any((path / "speech_tokenizer").glob("*.safetensors")):
            missing.append("speech_tokenizer/*.safetensors")
        if missing:
            raise ResourceNotFoundError(
                "Qwen3-TTS MLX snapshot is incomplete",
                details={"model_id": QWEN3_TTS_MODEL_ID, "missing": missing},
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_tts":
            raise ResourceIntegrityError(
                "Qwen3-TTS config has an unsupported model_type",
                details={"model_type": config.get("model_type")},
            )


class Qwen3TtsResourceProvider:
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
        raise UnsupportedRuntimeError(f"Qwen3-TTS conversion to {target_format} is not implemented")

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
    ) -> Qwen3TtsResourceResolver:
        if variant != QWEN3_TTS_VARIANT:
            raise ResourceNotFoundError(f"Qwen3-TTS variant is not registered: {variant}")
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError("Conflicting Qwen3-TTS variant values were provided")
        return Qwen3TtsResourceResolver(
            self._source,
            Qwen3TtsInstanceConfig.model_validate(normalized | {"variant": variant}),
        )
