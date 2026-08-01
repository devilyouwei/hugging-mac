"""Audio8-TTS ONNX INT4 resource resolution and lifecycle."""

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
    AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
    AUDIO8_TTS_ONNX_INT4_MODEL_ID,
    AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES,
    AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES,
    AUDIO8_TTS_ONNX_INT4_REVISION,
    AUDIO8_TTS_ONNX_INT4_VARIANT,
    Audio8TtsOnnxInt4InstanceConfig,
)


class Audio8TtsOnnxInt4ResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: Audio8TtsOnnxInt4InstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
    ) -> None:
        self._source = source
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
        if runtime not in {None, "onnx"}:
            raise UnsupportedRuntimeError(
                f"Audio8-TTS ONNX INT4 does not support runtime {runtime}",
                details={"model_id": AUDIO8_TTS_ONNX_INT4_MODEL_ID, "runtime": runtime},
            )
        root = self._model_root().expanduser().resolve(strict=False)
        target = (
            root if runtime is None else (self._config.source_path or self._default_source_path())
        )
        resolved = target.expanduser().resolve(strict=False)
        if resolved != root and not resolved.is_relative_to(root):
            raise ResourceIntegrityError(
                "Refusing to delete Audio8-TTS ONNX resources outside the model directory",
                details={"model_id": AUDIO8_TTS_ONNX_INT4_MODEL_ID},
            )
        await asyncio.to_thread(self._delete_path, resolved)

    def status(self) -> ModelResourceStatus:
        path = self._config.source_path or self._default_source_path()
        available = self._snapshot_files_exist(path)
        return ModelResourceStatus(
            model_id=AUDIO8_TTS_ONNX_INT4_MODEL_ID,
            revision=AUDIO8_TTS_ONNX_INT4_REVISION,
            variant=self._config.variant,
            artifacts=(
                ModelArtifactStatus(
                    artifact_id="source",
                    format=ArtifactFormat.ONNX.value,
                    runtime="onnx",
                    available=available,
                    size_bytes=directory_size(path) if available else None,
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "onnx" / "model"

    def _model_root(self) -> Path:
        return (
            self._config.model_home
            / "audio8"
            / "audio8-tts-preview-0.6b-onnx-int4"
            / self._source.revision
            / self._config.variant
        )

    @staticmethod
    def _snapshot_files_exist(path: Path) -> bool:
        return path.is_dir() and all(
            (path / name).is_file() for name in AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES
        )

    def _validate_snapshot(self, path: Path) -> str:
        if not path.is_dir():
            raise ResourceNotFoundError(
                "Audio8-TTS ONNX INT4 model is not downloaded",
                details={"model_id": AUDIO8_TTS_ONNX_INT4_MODEL_ID, "artifact_id": "source"},
            )
        missing = [
            name for name in AUDIO8_TTS_ONNX_INT4_REQUIRED_FILES if not (path / name).is_file()
        ]
        if missing:
            raise ResourceNotFoundError(
                "Audio8-TTS ONNX INT4 snapshot is incomplete",
                details={"model_id": AUDIO8_TTS_ONNX_INT4_MODEL_ID, "missing": missing},
            )
        manifest_path = path / "runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "model_family": "audio8_tts",
            "default_precision": "int4",
            "codec_sample_rate": 44100,
            "num_codebooks": 10,
            "model_fingerprint": AUDIO8_TTS_ONNX_INT4_FINGERPRINT,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ResourceIntegrityError(
                "Audio8-TTS ONNX runtime manifest is incompatible",
                details={"mismatches": mismatches},
            )
        registration_present = [
            (path / name).is_file() for name in AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES
        ]
        if any(registration_present) and not all(registration_present):
            raise ResourceIntegrityError(
                "Audio8-TTS ONNX voice-registration files are incomplete",
                details={
                    "missing": [
                        name
                        for name in AUDIO8_TTS_ONNX_INT4_REGISTRATION_FILES
                        if not (path / name).is_file()
                    ]
                },
            )
        if all(registration_present):
            registration = json.loads(
                (path / "registration/registration_manifest.json").read_text(encoding="utf-8")
            )
            if registration.get("model_fingerprint") != AUDIO8_TTS_ONNX_INT4_FINGERPRINT:
                raise ResourceIntegrityError(
                    "Audio8-TTS ONNX registration model fingerprint mismatch"
                )
        return file_sha256(manifest_path)

    @staticmethod
    def _delete_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class Audio8TtsOnnxInt4ResourceProvider:
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
        del options, overwrite
        raise UnsupportedRuntimeError(
            f"Audio8-TTS ONNX INT4 conversion to {target_format} is not implemented",
            details={"model_id": AUDIO8_TTS_ONNX_INT4_MODEL_ID, "variant": variant},
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
    ) -> Audio8TtsOnnxInt4ResourceResolver:
        if variant != AUDIO8_TTS_ONNX_INT4_VARIANT:
            raise ResourceNotFoundError(
                f"Audio8-TTS ONNX INT4 variant is not registered: {variant}",
                details={"supported_variants": [AUDIO8_TTS_ONNX_INT4_VARIANT]},
            )
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting Audio8-TTS ONNX INT4 variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = Audio8TtsOnnxInt4InstanceConfig.model_validate(normalized | {"variant": variant})
        return Audio8TtsOnnxInt4ResourceResolver(self._source, config)
