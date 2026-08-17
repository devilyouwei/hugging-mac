"""Downloaded ONNX source and converted Core ML resource lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
    UrlArchiveSource,
)

from .config import (
    MEDIAPIPE_HAND_DETECTION_MODEL_ID,
    MEDIAPIPE_HAND_DETECTION_REVISION,
    MediaPipeHandDetectionInstanceConfig,
)
from .converter import MediaPipeHandDetectionConverter

_REQUIRED = (
    "hand_detector.onnx",
    "hand_detector.data",
    "hand_landmark_detector.onnx",
    "hand_landmark_detector.data",
    "metadata.json",
)


class MediaPipeHandDetectionResourceResolver:
    def __init__(
        self,
        source: UrlArchiveSource,
        config: MediaPipeHandDetectionInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: MediaPipeHandDetectionConverter | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter or MediaPipeHandDetectionConverter()

    @property
    def model_root(self) -> Path:
        return (
            self._config.model_home
            / "qualcomm"
            / "mediapipe-hand-detection"
            / MEDIAPIPE_HAND_DETECTION_REVISION
            / self._config.variant
        )

    @property
    def source_path(self) -> Path:
        if self._config.source_path is not None:
            return self._config.source_path
        if self._config.runtime == "onnx" and self._config.artifact_path is not None:
            return self._config.artifact_path
        return self.model_root / "onnx"

    @property
    def coreml_path(self) -> Path:
        if self._config.runtime == "coreml" and self._config.artifact_path is not None:
            return self._config.artifact_path
        return self.model_root / "coreml"

    async def resolve_source(self) -> ResolvedResource:
        missing = [name for name in _REQUIRED if not (self.source_path / name).is_file()]
        if missing:
            raise ResourceNotFoundError(
                "MediaPipe Hand Detection ONNX source is not downloaded",
                details={"missing": missing},
            )
        return ResolvedResource(
            path=self.source_path,
            source=self._source,
            digest=directory_sha256(self.source_path),
            size_bytes=directory_size(self.source_path),
        )

    async def resolve_coreml(self) -> ResolvedResource:
        path = self.coreml_path
        required = (
            path / "hand_detector.mlpackage" / "Manifest.json",
            path / "hand_landmark_detector.mlpackage" / "Manifest.json",
        )
        if not path.is_dir() or not all(item.is_file() for item in required):
            raise ResourceNotFoundError(
                "MediaPipe Hand Detection Core ML detector and landmark artifacts "
                "have not both been converted",
                details={"missing": [item.parent.name for item in required if not item.is_file()]},
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=directory_sha256(path),
            size_bytes=directory_size(path),
        )

    async def download(self, *, overwrite: bool = False) -> ResolvedResource:
        if self._config.source_path is not None or (
            self._config.runtime == "onnx" and self._config.artifact_path is not None
        ):
            if overwrite:
                raise ResourceIntegrityError(
                    "Cannot overwrite an explicit MediaPipe Hand Detection source path"
                )
            return await self.resolve_source()
        if overwrite or not self.source_path.exists():
            await self._downloader.download(self._source, self.source_path, overwrite=overwrite)
        return await self.resolve_source()

    async def convert_coreml(self, *, overwrite: bool = False) -> None:
        if self.coreml_path.exists() and not overwrite:
            await self.resolve_coreml()
            return
        source = await self.resolve_source()
        await self._converter.convert(
            ConversionRequest(
                model_id=MEDIAPIPE_HAND_DETECTION_MODEL_ID,
                model_revision=MEDIAPIPE_HAND_DETECTION_REVISION,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.ONNX,
                target_format=ArtifactFormat.COREML,
                output_path=self.coreml_path,
                options={"precision": self._config.coreml_precision},
                overwrite=overwrite,
            )
        )

    def status(self) -> ModelResourceStatus:
        source_available = all((self.source_path / name).is_file() for name in _REQUIRED)
        coreml_available = all(
            (self.coreml_path / name / "Manifest.json").is_file()
            for name in ("hand_detector.mlpackage", "hand_landmark_detector.mlpackage")
        )
        return ModelResourceStatus(
            model_id=MEDIAPIPE_HAND_DETECTION_MODEL_ID,
            revision=MEDIAPIPE_HAND_DETECTION_REVISION,
            variant=self._config.variant,
            artifacts=(
                _status(
                    "onnx-float",
                    ArtifactFormat.ONNX,
                    self.source_path,
                    source_available,
                    runtime="onnx",
                ),
                _status(
                    "coreml-fp32",
                    ArtifactFormat.COREML,
                    self.coreml_path,
                    coreml_available,
                    runtime="coreml",
                    provisioning="convert",
                ),
            ),
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime is None:
            targets = (self.model_root,)
        elif runtime == "onnx":
            targets = (self.source_path,)
        elif runtime == "coreml":
            targets = (self.coreml_path,)
        else:
            raise UnsupportedRuntimeError(f"Unsupported runtime: {runtime}")
        root = self.model_root.expanduser().resolve(strict=False)
        for target in targets:
            resolved = target.expanduser().resolve(strict=False)
            if resolved != root and not resolved.is_relative_to(root):
                raise ResourceIntegrityError(
                    "Refusing to delete MediaPipe Hand Detection resources outside model root"
                )
            await asyncio.to_thread(shutil.rmtree, resolved, True)
        _prune(root)


class MediaPipeHandDetectionResourceProvider:
    def __init__(self, source: UrlArchiveSource) -> None:
        self._source = source

    def _resolver(
        self,
        variant: str,
        options: Mapping[str, object] | None,
        *,
        runtime: str | None = None,
    ) -> MediaPipeHandDetectionResourceResolver:
        normalized = dict(options or {})
        if runtime is not None:
            normalized["runtime"] = runtime
        return MediaPipeHandDetectionResourceResolver(
            self._source,
            MediaPipeHandDetectionInstanceConfig.model_validate(normalized | {"variant": variant}),
        )

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options, runtime="onnx")
        await resolver.download(overwrite=overwrite)
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
                f"MediaPipe Hand Detection cannot convert to {target_format}"
            )
        resolver = self._resolver(variant, options, runtime="coreml")
        await resolver.convert_coreml(overwrite=overwrite)
        return resolver.status()

    async def delete(
        self,
        variant: str,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        resolver = self._resolver(variant, options, runtime=runtime)
        await resolver.delete(runtime=runtime)
        return resolver.status()


def _status(
    artifact_id: str,
    artifact_format: ArtifactFormat,
    path: Path,
    available: bool,
    *,
    runtime: str,
    provisioning: Literal["download", "convert"] = "download",
) -> ModelArtifactStatus:
    return ModelArtifactStatus(
        artifact_id=artifact_id,
        format=artifact_format.value,
        runtime=runtime,
        provisioning=provisioning,
        available=available,
        size_bytes=directory_size(path) if available else None,
    )


def _prune(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        with contextlib.suppress(OSError):
            path.rmdir()
    with contextlib.suppress(OSError):
        root.rmdir()
