"""YOLOv8-specific resource resolution and lifecycle hooks."""

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
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
)

from .config import YoloV8InstanceConfig
from .converter import YoloV8Converter


class YoloV8ResourceResolver:
    def __init__(
        self,
        source: HuggingFaceSource,
        config: YoloV8InstanceConfig,
        manifest: ModelManifest,
        artifacts: Mapping[str, ModelArtifact],
        *,
        downloader: ResourceDownloader | None = None,
        converter: YoloV8Converter | None = None,
    ) -> None:
        self._source = source
        self._config = config
        self._manifest = manifest
        self._artifacts = dict(artifacts)
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter

    async def resolve_source(self) -> ResolvedResource:
        path = self._config.source_path or self._default_source_path()
        if not path.exists():
            raise ResourceNotFoundError(
                "YOLOv8 source model is not downloaded",
                details={
                    "model_id": self._manifest.model_id,
                    "variant": self._config.variant,
                    "artifact_id": "source",
                },
            )
        if not path.is_file():
            raise ResourceNotFoundError(f"YOLOv8 source is not a file: {path}")
        digest = file_sha256(path)
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
                    "model_id": self._manifest.model_id,
                    "variant": self._config.variant,
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

    async def resolve_onnx(self) -> ResolvedResource:
        path = self._config.artifact_path or self._default_onnx_path()
        if not path.is_file():
            raise ResourceNotFoundError(
                "YOLOv8 ONNX artifact has not been converted",
                details={
                    "model_id": self._manifest.model_id,
                    "variant": self._config.variant,
                    "artifact_id": "onnx",
                },
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=file_sha256(path),
            size_bytes=path.stat().st_size,
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
        await self.convert(ArtifactFormat.COREML, overwrite=overwrite)

    async def convert(
        self,
        target_format: ArtifactFormat,
        *,
        overwrite: bool = False,
    ) -> None:
        if target_format is ArtifactFormat.COREML:
            path = self._config.artifact_path or self._default_coreml_path()
            resolver = self.resolve_coreml
        elif target_format is ArtifactFormat.ONNX:
            path = self._config.artifact_path or self._default_onnx_path()
            resolver = self.resolve_onnx
        else:
            raise UnsupportedRuntimeError(f"YOLOv8 does not support {target_format}")
        if path.exists() and not overwrite:
            await resolver()
            return
        source = await self.resolve_source()
        if self._converter is None:
            raise ResourceIntegrityError("YOLOv8 converter is not configured")
        await self._converter.convert(
            ConversionRequest(
                model_id=self._manifest.model_id,
                model_revision=self._source.revision,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.PYTORCH,
                target_format=target_format,
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
        elif runtime == "onnx":
            targets = (self._config.artifact_path or self._default_onnx_path(),)
        else:
            raise UnsupportedRuntimeError(
                f"YOLOv8 does not support runtime {runtime}",
                details={
                    "model_id": self._manifest.model_id,
                    "variant": self._config.variant,
                    "runtime": runtime,
                },
            )
        await asyncio.to_thread(self._delete_targets, targets)

    def status(self) -> ModelResourceStatus:
        source_path = self._config.source_path or self._default_source_path()
        coreml_path = self._config.artifact_path or self._default_coreml_path()
        onnx_path = self._config.artifact_path or self._default_onnx_path()
        return ModelResourceStatus(
            model_id=self._manifest.model_id,
            revision=self._manifest.revision,
            variant=self._config.variant,
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
                _artifact_status(
                    "onnx",
                    ArtifactFormat.ONNX,
                    onnx_path,
                    runtime="onnx",
                    provisioning="convert",
                ),
            ),
        )

    def _default_source_path(self) -> Path:
        return self._artifacts["source"].resolve(self._config.model_home)

    def _default_coreml_path(self) -> Path:
        return self._artifacts["coreml"].resolve(self._config.model_home)

    def _default_onnx_path(self) -> Path:
        return self._artifacts["onnx"].resolve(self._config.model_home)

    def _model_root(self) -> Path:
        return self._artifacts["source"].storage_path(self._config.model_home).parents[1]

    def _delete_targets(self, targets: tuple[Path, ...]) -> None:
        root = self._model_root().expanduser().resolve(strict=False)
        for target in targets:
            resolved = target.expanduser().resolve(strict=False)
            if resolved != root and not resolved.is_relative_to(root):
                raise ResourceIntegrityError(
                    "Refusing to delete a YOLOv8 artifact outside its model directory",
                    details={
                        "model_id": self._manifest.model_id,
                        "variant": self._config.variant,
                    },
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
        sources: Mapping[str, HuggingFaceSource],
        manifest: ModelManifest,
        artifacts: tuple[ModelArtifact, ...],
        converter: YoloV8Converter,
    ) -> None:
        self._sources = dict(sources)
        self._manifest = manifest
        self._artifacts = artifacts
        self._converter = converter

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
        if target_format not in {ArtifactFormat.COREML, ArtifactFormat.ONNX}:
            raise UnsupportedRuntimeError(
                f"YOLOv8 resource provider does not support {target_format} yet",
                details={
                    "model_id": self._manifest.model_id,
                    "variant": variant,
                    "target_format": target_format,
                },
            )
        resolver = self._resolver(variant, options)
        await resolver.convert(target_format, overwrite=overwrite)
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
    ) -> YoloV8ResourceResolver:
        try:
            if variant not in self._sources:
                raise KeyError(variant)
            typed_variant = variant
            source = self._sources[variant]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"YOLOv8 variant is not registered: {variant}",
                details={
                    "model_id": self._manifest.model_id,
                    "variant": variant,
                    "supported_variants": sorted(self._sources),
                },
            ) from error
        normalized = dict(options or {})
        option_variant = normalized.get("variant")
        if option_variant not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting YOLOv8 variant values were provided",
                details={"variant": variant, "options_variant": option_variant},
            )
        config = YoloV8InstanceConfig.model_validate(normalized | {"variant": typed_variant})
        artifacts = {
            artifact.artifact_id: artifact
            for artifact in self._artifacts
            if artifact.variant == variant
        }
        return YoloV8ResourceResolver(
            source, config, self._manifest, artifacts, converter=self._converter
        )


def _artifact_status(
    artifact_id: str,
    artifact_format: ArtifactFormat,
    path: Path,
    *,
    runtime: str | None,
    provisioning: Literal["download", "convert"] = "download",
) -> ModelArtifactStatus:
    available = path.is_dir() if artifact_format is ArtifactFormat.COREML else path.is_file()
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
        with contextlib.suppress(OSError):
            directory.rmdir()
    with contextlib.suppress(OSError):
        root.rmdir()
