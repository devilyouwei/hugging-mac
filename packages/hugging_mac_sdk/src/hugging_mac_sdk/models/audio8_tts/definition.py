"""Audio8-TTS definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8TtsInstanceConfig
from .instance import Audio8TtsInstance
from .resources import Audio8TtsResourceProvider, Audio8TtsResourceResolver
from .torch import TorchAudio8TtsEngine

AUDIO8_TTS_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_MANIFEST = AUDIO8_TTS_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = AUDIO8_TTS_MANIFEST.get_variant("preview").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Audio8-TTS needs one Hugging Face snapshot source")
    return resources[0]


def _create_pytorch(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsInstanceConfig.model_validate(
        options | {"runtime": "pytorch"}
    )
    resources = Audio8TtsResourceResolver(_source(), config)
    return Audio8TtsInstance(config, TorchAudio8TtsEngine(config, resources))


AUDIO8_TTS_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_MANIFEST,
    runtime_factories={"pytorch": _create_pytorch},
    artifacts=AUDIO8_TTS_CONFIG.artifacts,
    resource_provider=Audio8TtsResourceProvider(_source()),
)


def register_audio8_tts(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    """Register the pinned Audio8-TTS Preview model."""

    models.register(AUDIO8_TTS_DEFINITION, replace=replace)
    return AUDIO8_TTS_DEFINITION
