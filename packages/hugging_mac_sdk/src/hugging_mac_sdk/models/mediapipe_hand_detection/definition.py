"""Unified MediaPipe palm detector and optional hand landmarker registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import UrlArchiveSource

from .config import MediaPipeHandDetectionInstanceConfig
from .converter import MediaPipeHandDetectionConverter
from .coreml import CoreMlMediaPipeHandDetectionEngine
from .instance import MediaPipeHandDetectionInstance
from .onnx import OnnxMediaPipeHandDetectionEngine
from .resources import (
    MediaPipeHandDetectionResourceProvider,
    MediaPipeHandDetectionResourceResolver,
)

MEDIAPIPE_HAND_DETECTION_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
MEDIAPIPE_HAND_DETECTION_MANIFEST = MEDIAPIPE_HAND_DETECTION_CONFIG.manifest


def _source() -> UrlArchiveSource:
    resources = MEDIAPIPE_HAND_DETECTION_MANIFEST.get_variant("float").resources
    if len(resources) != 1 or not isinstance(resources[0], UrlArchiveSource):
        raise TypeError("MediaPipe Hand Detection requires one URL archive source")
    return resources[0]


def _create_onnx(options: dict[str, object]) -> MediaPipeHandDetectionInstance:
    config = MediaPipeHandDetectionInstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = MediaPipeHandDetectionResourceResolver(_source(), config)
    return MediaPipeHandDetectionInstance(
        config, OnnxMediaPipeHandDetectionEngine(config, resources)
    )


def _create_coreml(options: dict[str, object]) -> MediaPipeHandDetectionInstance:
    normalized = dict(options)
    requested_device = normalized.pop("device", None)
    if requested_device is not None:
        normalized["compute_units"] = requested_device
    config = MediaPipeHandDetectionInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = MediaPipeHandDetectionResourceResolver(_source(), config)
    return MediaPipeHandDetectionInstance(
        config, CoreMlMediaPipeHandDetectionEngine(config, resources)
    )


MEDIAPIPE_HAND_DETECTION_DEFINITION = ModelDefinition(
    manifest=MEDIAPIPE_HAND_DETECTION_MANIFEST,
    runtime_factories={"onnx": _create_onnx, "coreml": _create_coreml},
    artifacts=MEDIAPIPE_HAND_DETECTION_CONFIG.artifacts,
    converter_ids=("qualcomm.mediapipe-hand-detection",),
    resource_provider=MediaPipeHandDetectionResourceProvider(_source()),
)


def register_mediapipe_hand_detection(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(MediaPipeHandDetectionConverter(), replace=replace)
    models.register(MEDIAPIPE_HAND_DETECTION_DEFINITION, replace=replace)
    return MEDIAPIPE_HAND_DETECTION_DEFINITION
