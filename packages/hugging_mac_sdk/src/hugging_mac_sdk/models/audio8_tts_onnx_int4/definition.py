"""Audio8-TTS ONNX INT4 definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8TtsOnnxInt4InstanceConfig
from .instance import Audio8TtsOnnxInt4Instance
from .onnx import OnnxAudio8TtsInt4Engine
from .resources import (
    Audio8TtsOnnxInt4ResourceProvider,
    Audio8TtsOnnxInt4ResourceResolver,
)

AUDIO8_TTS_ONNX_INT4_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_ONNX_INT4_MANIFEST = AUDIO8_TTS_ONNX_INT4_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = AUDIO8_TTS_ONNX_INT4_MANIFEST.get_variant("int4").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Audio8-TTS ONNX INT4 needs one Hugging Face snapshot source")
    return resources[0]


def _create_onnx(options: dict[str, object]) -> Audio8TtsOnnxInt4Instance:
    config = Audio8TtsOnnxInt4InstanceConfig.model_validate(options | {"runtime": "onnx"})
    resources = Audio8TtsOnnxInt4ResourceResolver(_source(), config)
    return Audio8TtsOnnxInt4Instance(
        config,
        OnnxAudio8TtsInt4Engine(config, resources),
    )


AUDIO8_TTS_ONNX_INT4_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_ONNX_INT4_MANIFEST,
    runtime_factories={"onnx": _create_onnx},
    artifacts=AUDIO8_TTS_ONNX_INT4_CONFIG.artifacts,
    resource_provider=Audio8TtsOnnxInt4ResourceProvider(_source()),
)


def register_audio8_tts_onnx_int4(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    """Register the pinned official Audio8-TTS ONNX INT4 model."""

    models.register(AUDIO8_TTS_ONNX_INT4_DEFINITION, replace=replace)
    return AUDIO8_TTS_ONNX_INT4_DEFINITION
