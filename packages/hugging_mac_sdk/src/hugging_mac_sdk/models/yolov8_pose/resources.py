"""YOLOv8 Pose-specific resource resolution and lifecycle hooks."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size, file_sha256
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ModelArtifactStatus,
    ModelResourceStatus,
    ResolvedResource,
    ResourceSource,
)

from .config import (
    YOLOV8_POSE_MODEL_ID,
    YOLOV8_POSE_REVISION,
    YoloV8PoseInstanceConfig,
)
from .converter import YoloV8PoseConverter


class _YoloTaskConfig(Protocol):
    @property
    def variant(self) -> str: ...

    @property
    def model_home(self) -> Path: ...

    @property
    def source_path(self) -> Path | None: ...

    @property
    def artifact_path(self) -> Path | None: ...

    @property
    def hf_token(self) -> str | None: ...


class _YoloTaskResourceResolver:
    """Resolve one task/variant without performing implicit network work."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        source: ResourceSource,
        config: _YoloTaskConfig,
        downloader: ResourceDownloader | None = None,
        converter: ModelConverter,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._source = source
        self._config = config
        self._downloader = downloader or ResourceDownloader()
        self._converter = converter

    async def resolve_source(self) -> ResolvedResource:
        path = self.source_path
        if not path.is_file():
            raise ResourceNotFoundError(
                "YOLO source model is not downloaded",
                details=self._details("source"),
            )
        digest = file_sha256(path)
        expected = getattr(self._source, "expected_sha256", None)
        if expected is not None and digest.lower() != expected.lower():
            raise ResourceIntegrityError(
                f"Source SHA-256 mismatch: {path}",
                details=self._details("source") | {"expected": expected, "actual": digest},
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=digest,
            size_bytes=path.stat().st_size,
        )

    async def resolve_coreml(self) -> ResolvedResource:
        path = self.coreml_path
        if not path.is_dir():
            raise ResourceNotFoundError(
                "YOLO Core ML artifact has not been converted",
                details=self._details("coreml"),
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=directory_sha256(path),
            size_bytes=directory_size(path),
        )

    async def resolve_onnx(self) -> ResolvedResource:
        path = self.onnx_path
        if not path.is_file():
            raise ResourceNotFoundError(
                "YOLO ONNX artifact has not been converted",
                details=self._details("onnx"),
            )
        return ResolvedResource(
            path=path,
            source=self._source,
            digest=file_sha256(path),
            size_bytes=path.stat().st_size,
        )

    async def download_source(self, *, overwrite: bool = False) -> ResolvedResource:
        if self.source_path.exists() and not overwrite:
            return await self.resolve_source()
        return await self._downloader.download(
            self._source,
            self.source_path,
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
            path = self.coreml_path
            resolver = self.resolve_coreml
        elif target_format is ArtifactFormat.ONNX:
            path = self.onnx_path
            resolver = self.resolve_onnx
        else:
            raise UnsupportedRuntimeError(
                f"{self._model_id} does not support {target_format}",
                details=self._details() | {"target_format": target_format},
            )
        if path.exists() and not overwrite:
            await resolver()
            return
        source = await self.resolve_source()
        await self._converter.convert(
            ConversionRequest(
                model_id=self._model_id,
                model_revision=self._model_revision,
                variant=self._config.variant,
                source=source,
                source_format=ArtifactFormat.PYTORCH,
                target_format=target_format,
                output_path=path,
                overwrite=overwrite,
            )
        )

    async def delete(self, *, runtime: str | None = None) -> None:
        if runtime is None:
            targets = (self.model_root,)
        elif runtime == "pytorch-mps":
            targets = (self.source_path,)
        elif runtime == "coreml":
            targets = (self.coreml_path,)
        elif runtime == "onnx":
            targets = (self.onnx_path,)
        else:
            raise UnsupportedRuntimeError(
                f"{self._model_id} does not support runtime {runtime}",
                details=self._details() | {"runtime": runtime},
            )
        await asyncio.to_thread(self._delete_targets, targets)

    def status(self) -> ModelResourceStatus:
        return ModelResourceStatus(
            model_id=self._model_id,
            revision=self._model_revision,
            variant=self._config.variant,
            artifacts=(
                _artifact_status("source", ArtifactFormat.PYTORCH, self.source_path, "pytorch-mps"),
                _artifact_status(
                    "coreml",
                    ArtifactFormat.COREML,
                    self.coreml_path,
                    "coreml",
                    provisioning="convert",
                ),
                _artifact_status(
                    "onnx",
                    ArtifactFormat.ONNX,
                    self.onnx_path,
                    "onnx",
                    provisioning="convert",
                ),
            ),
        )

    @property
    def model_root(self) -> Path:
        return self._config.model_home.joinpath(
            *self._model_id.split("/"), self._model_revision, self._config.variant
        )

    @property
    def source_path(self) -> Path:
        return self._config.source_path or self.model_root / "source" / self._filename

    @property
    def coreml_path(self) -> Path:
        return (
            self._config.artifact_path
            or self.model_root / "coreml" / f"{Path(self._filename).stem}.mlpackage"
        )

    @property
    def onnx_path(self) -> Path:
        return (
            self._config.artifact_path
            or self.model_root / "onnx" / f"{Path(self._filename).stem}.onnx"
        )

    @property
    def _filename(self) -> str:
        if isinstance(self._source, HuggingFaceSource):
            if self._source.filename is None:
                raise ResourceIntegrityError(
                    "A YOLO Hugging Face source must identify one file",
                    details=self._details("source"),
                )
            return self._source.filename
        return Path(str(self._source.url)).name

    def _details(self, artifact_id: str | None = None) -> dict[str, str]:
        details = {"model_id": self._model_id, "variant": self._config.variant}
        if artifact_id is not None:
            details["artifact_id"] = artifact_id
        return details

    def _delete_targets(self, targets: tuple[Path, ...]) -> None:
        root = self.model_root.expanduser().resolve(strict=False)
        for target in targets:
            resolved = target.expanduser().resolve(strict=False)
            if resolved != root and not resolved.is_relative_to(root):
                raise ResourceIntegrityError(
                    "Refusing to delete an artifact outside its model directory",
                    details=self._details(),
                )
            if resolved.is_symlink() or resolved.is_file():
                resolved.unlink(missing_ok=True)
            elif resolved.is_dir():
                shutil.rmtree(resolved)
        _prune_empty_directories(root)


class _YoloTaskResourceProvider:
    """Variant-aware resource provider reusable by Pose and Seg model packages."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        sources: Mapping[str, ResourceSource],
        config_factory: Callable[[dict[str, object]], _YoloTaskConfig],
        converter_factory: Callable[[], ModelConverter],
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._sources = dict(sources)
        self._config_factory = config_factory
        self._converter_factory = converter_factory

    async def status(
        self, variant: str, options: Mapping[str, object] | None = None
    ) -> ModelResourceStatus:
        return self._resolver(variant, options).status()

    async def download_source(
        self, variant: str, options: Mapping[str, object] | None = None, *, overwrite: bool = False
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
                f"{self._model_id} does not support {target_format} yet",
                details={
                    "model_id": self._model_id,
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
        self, variant: str, options: Mapping[str, object] | None
    ) -> _YoloTaskResourceResolver:
        try:
            source = self._sources[variant]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"Model variant is not registered: {variant}",
                details={
                    "model_id": self._model_id,
                    "variant": variant,
                    "supported_variants": sorted(self._sources),
                },
            ) from error
        normalized = dict(options or {})
        if normalized.get("variant") not in {None, variant}:
            raise ResourceIntegrityError(
                "Conflicting model variant values were provided",
                details={"variant": variant, "options_variant": normalized["variant"]},
            )
        config = self._config_factory(normalized | {"variant": variant})
        return _YoloTaskResourceResolver(
            model_id=self._model_id,
            model_revision=self._model_revision,
            source=source,
            config=config,
            converter=self._converter_factory(),
        )


def _artifact_status(
    artifact_id: str,
    artifact_format: ArtifactFormat,
    path: Path,
    runtime: str,
    *,
    provisioning: Literal["download", "convert"] = "download",
) -> ModelArtifactStatus:
    available = path.is_dir() if artifact_format is ArtifactFormat.COREML else path.is_file()
    size = (
        path.stat().st_size
        if available and path.is_file()
        else directory_size(path)
        if available
        else None
    )
    return ModelArtifactStatus(
        artifact_id=artifact_id,
        format=artifact_format.value,
        runtime=runtime,
        provisioning=provisioning,
        available=available,
        size_bytes=size,
    )


def _prune_empty_directories(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        with contextlib.suppress(OSError):
            directory.rmdir()
    with contextlib.suppress(OSError):
        root.rmdir()


class YoloV8PoseResourceResolver(_YoloTaskResourceResolver):
    def __init__(
        self,
        source: ResourceSource,
        config: YoloV8PoseInstanceConfig,
        *,
        downloader: ResourceDownloader | None = None,
        converter: YoloV8PoseConverter | None = None,
    ) -> None:
        super().__init__(
            model_id=YOLOV8_POSE_MODEL_ID,
            model_revision=YOLOV8_POSE_REVISION,
            source=source,
            config=config,
            downloader=downloader,
            converter=converter or YoloV8PoseConverter(),
        )


class YoloV8PoseResourceProvider(_YoloTaskResourceProvider):
    def __init__(self, sources: Mapping[str, ResourceSource]) -> None:
        super().__init__(
            model_id=YOLOV8_POSE_MODEL_ID,
            model_revision=YOLOV8_POSE_REVISION,
            sources=sources,
            config_factory=lambda values: YoloV8PoseInstanceConfig.model_validate(values),
            converter_factory=YoloV8PoseConverter,
        )
