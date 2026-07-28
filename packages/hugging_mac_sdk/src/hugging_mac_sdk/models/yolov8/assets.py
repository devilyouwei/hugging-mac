"""Resolve pinned YOLOv8 source and converted runtime assets."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.models.yolov8.config import (
    YOLOV8N_FILENAME,
    YOLOV8N_SHA256,
    YoloV8InstanceConfig,
)
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)


class YoloV8AssetResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: YoloV8InstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: YoloV8Converter | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter or YoloV8Converter()

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if not path.exists():
            raise ResourceNotFoundError(
                "YOLOv8 source model is not downloaded",
                details={
                    "model_id": "ultralytics/yolov8n",
                    "artifact_id": "source",
                },
            )
        if not path.is_file():
            raise ResourceNotFoundError(f"YOLOv8 source is not a file: {path}")
        digest = file_sha256(path)
        if digest != YOLOV8N_SHA256:
            raise ResourceIntegrityError(
                f"YOLOv8 source SHA-256 mismatch: {path}",
                details={"expected": YOLOV8N_SHA256, "actual": digest},
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=path.stat().st_size,
        )

    async def resolve_coreml(self) -> ResolvedResource:
        path = self._config.artifact_path or self._default_coreml_path()
        if not path.exists():
            raise ResourceNotFoundError(
                "YOLOv8 Core ML artifact has not been converted",
                details={
                    "model_id": "ultralytics/yolov8n",
                    "artifact_id": "coreml",
                },
            )
        if not path.is_dir():
            raise ResourceNotFoundError(f"Core ML artifact is not a directory: {path}")
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=directory_sha256(path),
            size_bytes=directory_size(path),
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if path.exists() and not overwrite:
            return await self.resolve_source()
        return await self._downloader.download(
            self._source,
            path,
            overwrite=overwrite,
            token=self._config.hf_token,
        )

    async def convert_coreml(self, *, overwrite: bool = False) -> None:
        path = self._config.artifact_path or self._default_coreml_path()
        if path.exists() and not overwrite:
            await self.resolve_coreml()
            return
        source = await self.resolve_source()
        await self._converter.convert(
            ConversionRequest(
                model_id="ultralytics/yolov8n",
                model_revision=self._source.revision,
                source=source,
                source_format=ArtifactFormat.PYTORCH,
                target_format=ArtifactFormat.COREML,
                output_path=path,
                overwrite=overwrite,
            )
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        """Delete one runtime artifact or this revision's complete model directory."""

        if runtime is None:
            targets = (self._model_root(),)
        elif runtime == "pytorch-mps":
            targets = (self._config.source_path or self._default_source_path(),)
        elif runtime == "coreml":
            targets = (self._config.artifact_path or self._default_coreml_path(),)
        else:
            raise UnsupportedRuntimeError(
                f"YOLOv8 does not support runtime {runtime}",
                details={"model_id": "ultralytics/yolov8n", "runtime": runtime},
            )
        await asyncio.to_thread(self._delete_targets, targets)

    def status(self, *, revision: str) -> ModelResourceStatus:
        source_path = self._config.source_path or self._default_source_path()
        coreml_path = self._config.artifact_path or self._default_coreml_path()
        return ModelResourceStatus(
            model_id="ultralytics/yolov8n",
            revision=revision,
            artifacts=(
                _artifact_status(
                    "source",
                    ArtifactFormat.PYTORCH,
                    source_path,
                    runtime="pytorch-mps",
                ),
                _artifact_status(
                    "coreml",
                    ArtifactFormat.COREML,
                    coreml_path,
                    runtime="coreml",
                    provisioning="convert",
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._model_root() / "source" / YOLOV8N_FILENAME

    def _default_coreml_path(self) -> Path:
        return self._model_root() / "coreml" / "yolov8n.mlpackage"

    def _model_root(self) -> Path:
        return self._config.model_home / "ultralytics" / "yolov8n" / self._source.revision

    def _delete_targets(self, targets: tuple[Path, ...]) -> None:
        root = self._model_root().expanduser().resolve(strict=False)
        for target in targets:
            resolved = target.expanduser().resolve(strict=False)
            if resolved != root and not resolved.is_relative_to(root):
                raise ResourceIntegrityError(
                    "Refusing to delete a YOLOv8 artifact outside its model directory",
                    details={"model_id": "ultralytics/yolov8n"},
                )
            if resolved.is_symlink() or resolved.is_file():
                resolved.unlink(missing_ok=True)
            elif resolved.is_dir():
                shutil.rmtree(resolved)
        if root.exists():
            _prune_empty_directories(root)


class YoloV8ResourceProvider:
    def __init__(
        self,
        source: HuggingFaceSource,
        *,
        revision: str,
    ) -> None:
        self._source = source
        self._revision = revision

    async def status(
        self,
        options: Mapping[str, object] | None = None,
    ) -> ModelResourceStatus:
        return self._resolver(options).status(revision=self._revision)

    async def download_source(
        self,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        resolver = self._resolver(options)
        await resolver.download_source(overwrite=overwrite)
        return resolver.status(revision=self._revision)

    async def convert(
        self,
        target_format: ArtifactFormat,
        options: Mapping[str, object] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelResourceStatus:
        if target_format is not ArtifactFormat.COREML:
            raise UnsupportedRuntimeError(
                f"YOLOv8 resource provider does not support {target_format} yet",
                details={
                    "model_id": "ultralytics/yolov8n",
                    "target_format": target_format,
                },
            )
        resolver = self._resolver(options)
        await resolver.convert_coreml(overwrite=overwrite)
        return resolver.status(revision=self._revision)

    async def delete(
        self,
        options: Mapping[str, object] | None = None,
        *,
        runtime: str | None = None,
    ) -> ModelResourceStatus:
        resolver = self._resolver(options)
        await resolver.delete(runtime=runtime)
        return resolver.status(revision=self._revision)

    def _resolver(
        self,
        options: Mapping[str, object] | None,
    ) -> YoloV8AssetResolver:
        config = YoloV8InstanceConfig.model_validate(dict(options or {}))
        return YoloV8AssetResolver(self._source, config)


def _artifact_status(
    artifact_id: str,
    artifact_format: ArtifactFormat,
    path: Path,
    *,
    runtime: str | None,
    provisioning: Literal["download", "convert"] = "download",
) -> ModelArtifactStatus:
    available = path.is_file() if artifact_format is ArtifactFormat.PYTORCH else path.is_dir()
    size = None
    if available:
        size = path.stat().st_size if path.is_file() else directory_size(path)
    return ModelArtifactStatus(
        artifact_id=artifact_id,
        format=artifact_format.value,
        runtime=runtime,
        provisioning=provisioning,
        available=available,
        size_bytes=size,
    )


def _prune_empty_directories(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass
