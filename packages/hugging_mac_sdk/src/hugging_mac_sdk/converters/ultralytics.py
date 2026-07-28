"""Generic Ultralytics exporter for supported PyTorch model artifacts."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import DownloadError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import (
    directory_sha256,
    directory_size,
    file_sha256,
)
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)


class UltralyticsExportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=1, gt=0)
    dynamic: bool = False
    half: bool = False
    quantize: Literal[8, 16] | None = None
    nms: bool = False
    device: str = "cpu"
    simplify: bool = True
    opset: int | None = Field(default=None, ge=7)


class UltralyticsExportConverter(ModelConverter):
    """Generic converter for models understood by ``ultralytics.YOLO``."""

    @property
    def converter_id(self) -> str:
        return "ultralytics.export"

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.source_format is ArtifactFormat.PYTORCH
            and request.target_format in {ArtifactFormat.COREML, ArtifactFormat.ONNX}
            and request.source.path.is_file()
            and request.source.path.suffix == ".pt"
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(f"{self.converter_id} does not support this conversion")
        options = UltralyticsExportOptions.model_validate(request.options)
        output = request.output_path.expanduser().resolve()
        if output.exists() and not request.overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        staging_root = output.parent / f".{output.name}.convert-{uuid4().hex}"
        staging_root.mkdir()
        staged_source = staging_root / request.source.path.name
        shutil.copyfile(request.source.path, staged_source)

        try:
            exported = await asyncio.to_thread(
                self._export,
                staged_source,
                request.target_format,
                options,
            )
            if not exported.exists():
                raise DownloadError(f"Converter did not produce an artifact: {exported}")
            _commit_artifact(exported, output, overwrite=request.overwrite)
            digest, size = await asyncio.to_thread(_artifact_identity, output)
            return ConversionResult(
                path=output,
                format=request.target_format,
                digest=digest,
                size_bytes=size,
                converter_id=self.converter_id,
                model_id=request.model_id,
                model_revision=request.model_revision,
                variant=request.variant,
                source_digest=request.source.digest,
                options=options.model_dump(mode="json"),
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(staging_root)

    def _export(
        self,
        source: Path,
        target_format: ArtifactFormat,
        options: UltralyticsExportOptions,
    ) -> Path:
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "Ultralytics conversion requires: uv sync --package hugging-mac-sdk --extra yolo",
                cause=error,
            ) from error

        export_format = {
            ArtifactFormat.COREML: "coreml",
            ArtifactFormat.ONNX: "onnx",
        }[target_format]
        arguments: dict[str, Any] = {
            "format": export_format,
            "imgsz": options.imgsz,
            "batch": options.batch,
            "dynamic": options.dynamic,
            "half": options.half,
            "nms": options.nms,
            "device": options.device,
        }
        if options.quantize is not None:
            arguments["quantize"] = options.quantize
        if target_format is ArtifactFormat.ONNX:
            arguments["simplify"] = options.simplify
            if options.opset is not None:
                arguments["opset"] = options.opset

        model = ultralytics.YOLO(str(source))
        return Path(model.export(**arguments)).resolve()


def _artifact_identity(path: Path) -> tuple[str, int]:
    if path.is_dir():
        return directory_sha256(path), directory_size(path)
    return file_sha256(path), path.stat().st_size


def _commit_artifact(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    os.replace(staging, output)
