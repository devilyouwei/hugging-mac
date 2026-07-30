"""Shared export pipeline extended by self-contained YOLOv8 model packs."""

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


class YoloV8ExportOptions(BaseModel):
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


class YoloV8ExportConverter(ModelConverter):
    """Provide format export mechanics without owning checkpoint semantics."""

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
        options = YoloV8ExportOptions.model_validate(request.options)
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
        options: YoloV8ExportOptions,
    ) -> Path:
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "YOLO conversion requires PyTorch; install the yolo extra",
                cause=error,
            ) from error
        if options.nms:
            raise UnsupportedRuntimeError(
                "Native YOLO conversion exports raw predictions; NMS is applied by the model SDK"
            )
        model = self._load_checkpoint(source, torch)
        example = torch.zeros(options.batch, 3, options.imgsz, options.imgsz)
        task = _checkpoint_task(source)
        if target_format is ArtifactFormat.ONNX:
            return _export_onnx(source, model, example, task, options, torch)
        return _export_coreml(source, model, example, task, options, torch)

    def _load_checkpoint(self, source: Path, torch: Any) -> Any:
        """Implemented by each model pack using its private checkpoint module."""

        raise NotImplementedError


def _checkpoint_task(source: Path) -> Literal["detect", "pose", "segment"]:
    if "-pose" in source.stem:
        return "pose"
    if "-seg" in source.stem:
        return "segment"
    return "detect"


def _export_onnx(
    source: Path,
    model: Any,
    example: Any,
    task: str,
    options: YoloV8ExportOptions,
    torch: Any,
) -> Path:
    output = source.with_suffix(".onnx")
    output_names = ["predictions", "prototypes"] if task == "segment" else ["predictions"]
    dynamic_axes = None
    if options.dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "predictions": {0: "batch", 2: "anchors"},
        }
        if task == "segment":
            dynamic_axes["prototypes"] = {0: "batch", 2: "mask_height", 3: "mask_width"}
    try:
        torch.onnx.export(
            model,
            example,
            str(output),
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=options.opset or 17,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as error:
        raise UnsupportedRuntimeError(
            "Native ONNX export failed; ensure the onnx package is installed",
            cause=error,
        ) from error
    return output.resolve()


def _export_coreml(
    source: Path,
    model: Any,
    example: Any,
    task: str,
    options: YoloV8ExportOptions,
    torch: Any,
) -> Path:
    try:
        coremltools = importlib.import_module("coremltools")
    except ImportError as error:
        raise UnsupportedRuntimeError(
            "Core ML conversion requires coremltools on Apple Silicon",
            cause=error,
        ) from error
    output = source.with_suffix(".mlpackage")
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    outputs = [coremltools.TensorType(name="predictions")]
    if task == "segment":
        outputs.append(coremltools.TensorType(name="prototypes"))
    precision = (
        coremltools.precision.FLOAT16
        if options.half or options.quantize == 16
        else coremltools.precision.FLOAT32
    )
    try:
        converted = coremltools.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                coremltools.ImageType(
                    name="image",
                    shape=example.shape,
                    scale=1 / 255.0,
                )
            ],
            outputs=outputs,
            compute_precision=precision,
        )
        if options.quantize == 8:
            optimize = coremltools.optimize.coreml
            converted = optimize.palettize_weights(
                converted,
                config=optimize.OptimizationConfig(
                    global_config=optimize.OpPalettizerConfig(
                        mode="kmeans",
                        nbits=8,
                    )
                ),
            )
        converted.user_defined_metadata.update(
            {
                "hugging_mac_task": task,
                "hugging_mac_input_size": str(options.imgsz),
                "hugging_mac_raw_outputs": "true",
            }
        )
        converted.save(str(output))
    except Exception as error:
        raise UnsupportedRuntimeError("Native Core ML export failed", cause=error) from error
    return output.resolve()


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
