"""Model-specific ONNX to Core ML conversion."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.errors import ResourceNotFoundError, UnsupportedRuntimeError
from hugging_mac_sdk.resources.hashing import directory_sha256, directory_size
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult

from .config import MEDIAPIPE_HAND_DETECTION_MODEL_ID, MediaPipeHandDetectionCoreMlConfig


class MediaPipeHandDetectionConverter(ModelConverter):
    @property
    def converter_id(self) -> str:
        return "qualcomm.mediapipe-hand-detection"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, request: ConversionRequest) -> bool:
        source = request.source.path
        has_models = source.is_dir() and all(
            (source / name).is_file()
            for name in ("hand_detector.onnx", "hand_landmark_detector.onnx")
        )
        return (
            request.model_id == MEDIAPIPE_HAND_DETECTION_MODEL_ID
            and request.variant == "float"
            and request.source_format is ArtifactFormat.ONNX
            and request.target_format is ArtifactFormat.COREML
            and has_models
        )

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        if not self.supports(request):
            raise UnsupportedRuntimeError(f"{self.converter_id} does not support this conversion")
        options = MediaPipeHandDetectionCoreMlConfig.model_validate(request.options)
        output = request.output_path.expanduser().resolve()
        if output.exists() and not request.overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / f".{output.name}.convert-{uuid4().hex}"
        try:
            await asyncio.to_thread(
                self._convert,
                request.source.path,
                staging,
                options,
            )
            if not staging.is_dir():
                raise ResourceNotFoundError(
                    f"Core ML converter did not produce a package: {staging}"
                )
            _commit(staging, output, overwrite=request.overwrite)
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
        options: MediaPipeHandDetectionCoreMlConfig,
    ) -> None:
        onnx = _module("onnx", "Conversion requires onnx; install the vision extra")
        onnx2torch = _module(
            "onnx2torch", "Conversion requires onnx2torch; install the vision extra"
        )
        torch = _module("torch", "Conversion requires PyTorch; install the vision extra")
        coremltools = _module("coremltools", "Conversion requires coremltools on Apple Silicon")

        example = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        output.mkdir(parents=True)
        specifications = (
            ("hand_detector", ("box_coords", "box_scores")),
            ("hand_landmark_detector", ("scores", "lr", "landmarks")),
        )
        for name, output_names in specifications:
            graph = onnx.load(str(source / f"{name}.onnx"), load_external_data=True)
            _normalize_exported_graph(graph)
            model = onnx2torch.convert(graph).eval()
            with torch.inference_mode():
                traced = torch.jit.trace(model, example, strict=False)
            converted = coremltools.convert(
                traced,
                convert_to="mlprogram",
                inputs=[coremltools.TensorType(name="image", shape=example.shape, dtype=float)],
                outputs=[coremltools.TensorType(name=value) for value in output_names],
                compute_precision=coremltools.precision.FLOAT32,
                minimum_deployment_target=coremltools.target.macOS13,
            )
            converted.user_defined_metadata.update(
                {
                    "hugging_mac_model_id": MEDIAPIPE_HAND_DETECTION_MODEL_ID,
                    "hugging_mac_component": name,
                    "hugging_mac_task": "hand-detection",
                    "hugging_mac_input_size": "256",
                    "hugging_mac_raw_outputs": "true",
                    "hugging_mac_source_format": "onnx",
                }
            )
            converted.save(str(output / f"{name}.mlpackage"))


def _normalize_exported_graph(graph: Any) -> None:
    """Bridge Qualcomm's opset-21 export to onnx2torch's equivalent opset-17 handlers."""

    default_opset = next(item for item in graph.opset_import if not item.domain)
    if default_opset.version < 17:
        raise UnsupportedRuntimeError(
            f"Unsupported MediaPipe Hand ONNX opset: {default_opset.version}"
        )
    # Pad semantics used by this fixed graph are unchanged between opsets 17 and 21.
    default_opset.version = 17
    for node in graph.graph.node:
        if node.op_type == "Reshape":
            for attribute in node.attribute:
                if attribute.name == "allowzero":
                    # All shipped reshape constants are non-zero; onnx2torch does not
                    # implement allowzero=1 even though it has no effect for this graph.
                    attribute.i = 0


def _module(name: str, message: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise UnsupportedRuntimeError(message, cause=error) from error


def _commit(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Conversion output already exists: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    os.replace(staging, output)
