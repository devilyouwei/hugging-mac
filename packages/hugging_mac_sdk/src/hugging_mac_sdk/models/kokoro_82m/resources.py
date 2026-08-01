"""Kokoro-82M PyTorch resource resolution, status, and deletion."""

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
    KOKORO_82M_DEFAULT_VOICE_SHA256,
    KOKORO_82M_MODEL_ID,
    KOKORO_82M_REQUIRED_FILES,
    KOKORO_82M_REVISION,
    KOKORO_82M_VARIANT,
    KOKORO_82M_WEIGHT_SHA256,
    Kokoro82mInstanceConfig,
)


class Kokoro82mResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Kokoro82mInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader(timeout=1800)

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
                f"Kokoro-82M does not support runtime {runtime}",
                details={"model_id": KOKORO_82M_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        if runtime is None:
            target = root
        else:
            target = self._config.source_path or self._default_source_path()
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Kokoro-82M resources outside the model directory",
                details={"model_id": KOKORO_82M_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        source = self._config.source_path or self._default_source_path()
        source_available = self._snapshot_files_exist(source)
        return ModelResourceStatus(
            model_id=KOKORO_82M_MODEL_ID,
            revision=KOKORO_82M_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.PYTORCH.value,
                    runtime="pytorch-mps",
                    available=source_available,
                    size_bytes=directory_size(source) if source_available else None,
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "pytorch" / "model"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "hexgrad"
            / "kokoro-82m"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).is_file() for name in KOKORO_82M_REQUIRED_FILES
        )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "Kokoro-82M source model is not downloaded",
                details={"model_id": KOKORO_82M_MODEL_ID, "artifact_id": "source"},
            )
        missing = [
            name for name in KOKORO_82M_REQUIRED_FILES if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Kokoro-82M snapshot is incomplete",
                details={"model_id": KOKORO_82M_MODEL_ID, "missing": missing},
            )
        digest = file_sha256(path / "kokoro-v1_0.pth")
        if digest != KOKORO_82M_WEIGHT_SHA256:
            raise ResourceIntegrityError(
                "Kokoro-82M weight SHA-256 mismatch",
                details={"expected": KOKORO_82M_WEIGHT_SHA256, "actual": digest},
            )
        voice_digest = file_sha256(path / "voices" / "af_heart.pt")
        if voice_digest != KOKORO_82M_DEFAULT_VOICE_SHA256:
            raise ResourceIntegrityError(
                "Kokoro-82M default voice SHA-256 mismatch",
                details={
                    "expected": KOKORO_82M_DEFAULT_VOICE_SHA256,
                    "actual": voice_digest,
                },
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if "vocab" not in config or int(config.get("n_token", 0)) <= 0:
            raise ResourceIntegrityError("Kokoro-82M config is invalid")
        return digest

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Kokoro82mResourceProvider:
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
        raise UnsupportedRuntimeError("Kokoro-82M does not support conversion")

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
    ) -> Kokoro82mResourceResolver:
        if variant != KOKORO_82M_VARIANT:
            raise ResourceNotFoundError(
                f"Kokoro-82M variant is not registered: {variant}",
                details={"supported_variants": [KOKORO_82M_VARIANT]},
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting Kokoro-82M variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = Kokoro82mInstanceConfig.model_validate(
            normalized | {"variant": variant}
        )
        return Kokoro82mResourceResolver(self._source, config)
