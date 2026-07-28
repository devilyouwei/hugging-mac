"""Schemas for reproducible model artifact conversion."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hugging_mac_sdk.schemas.resources import ResolvedResource


class ArtifactFormat(StrEnum):
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    COREML = "coreml"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    MLX = "mlx"
    RKNN = "rknn"
    TFLITE = "tflite"
    OPENVINO = "openvino"


class ConversionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    source: ResolvedResource
    source_format: ArtifactFormat
    target_format: ArtifactFormat
    output_path: Path
    model_id: str | None = None
    model_revision: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False


class ConversionResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    format: ArtifactFormat
    digest: str
    size_bytes: int
    converter_id: str
    source_digest: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
