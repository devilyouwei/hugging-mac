"""Convert the pinned py-feat RetinaFace checkpoint to Core ML."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import shutil
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)

from .config import RETINAFACE_MODEL_ID
from .utils.checkpoint import load_retinaface_checkpoint


class RetinaFaceConversionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_size: int = Field(default=640, ge=320, le=1280, multiple_of=32)
    half: bool = True


class RetinaFaceConverter(ModelConverter):
    converter_id = "py-feat.retinaface"

    def supports(self, request: ConversionRequest) -> bool:
        return (
            request.model_id == RETINAFACE_MODEL_ID
            and request.source_format is ArtifactFormat.PYTORCH
            and request.target_format is ArtifactFormat.COREML
            and request.source.path.is_dir()
            and (request.source.path / "config.json").is_file()
            and (request.source.path / "mobilenet0.25_Final.pth").is_file()
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(f"{self.converter_id} does not support this conversion")
        options = RetinaFaceConversionOptions.model_validate(request.options)
        output = request.output_path.expanduser().resolve()
        if output.exists() and not request.overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / f".{output.name}.convert-{uuid4().hex}.mlpackage"
        try:
            await asyncio.to_thread(self._convert, request.source.path, staging, options)
            if output.exists():
                if output.is_dir():
                    shutil.rmtree(output)
                else:
                    output.unlink()
            os.replace(staging, output)
            return ConversionResult(
                path=output,
                format=ArtifactFormat.COREML,
                digest=directory_sha256(output),
                size_bytes=directory_size(output),
                converter_id=self.converter_id,
                model_id=request.model_id,
                model_revision=request.model_revision,
                variant=request.variant,
                source_digest=request.source.digest,
                options=options.model_dump(mode="json"),
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(staging)

    def _convert(
        self,
        source: Path,
        output: Path,
        options: RetinaFaceConversionOptions,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            coremltools = importlib.import_module("coremltools")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "RetinaFace conversion requires PyTorch and coremltools",
                cause=error,
            ) from error

        model = load_retinaface_checkpoint(source, torch)
        example = torch.zeros(1, 3, options.input_size, options.input_size)
        with torch.inference_mode():
            traced = torch.jit.trace(model, example, strict=False)
        precision = coremltools.precision.FLOAT16 if options.half else coremltools.precision.FLOAT32
        try:
            converted = coremltools.convert(
                traced,
                convert_to="mlprogram",
                inputs=[
                    coremltools.TensorType(
                        name="image",
                        shape=example.shape,
                    )
                ],
                outputs=[
                    coremltools.TensorType(name="locations"),
                    coremltools.TensorType(name="scores"),
                    coremltools.TensorType(name="landmarks"),
                ],
                compute_precision=precision,
            )
            converted.user_defined_metadata.update(
                {
                    "hugging_mac_model": RETINAFACE_MODEL_ID,
                    "hugging_mac_input_size": str(options.input_size),
                    "hugging_mac_raw_outputs": "true",
                }
            )
            converted.save(str(output))
        except Exception as error:
            raise UnsupportedRuntimeError(
                "RetinaFace Core ML conversion failed",
                cause=error,
            ) from error
