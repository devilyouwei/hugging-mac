"""Audio8-ASR definition, PyTorch factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8AsrInstanceConfig
from .instance import Audio8AsrInstance
from .resources import Audio8AsrResourceProvider, Audio8AsrResourceResolver
from .torch import TorchAudio8AsrEngine

AUDIO8_ASR_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_ASR_MANIFEST = AUDIO8_ASR_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = AUDIO8_ASR_MANIFEST.get_variant("base").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Audio8-ASR needs one Hugging Face snapshot source")
    return resources[0]


def _create_pytorch_mps(options: dict[str, object]) -> Audio8AsrInstance:
    config = Audio8AsrInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = Audio8AsrResourceResolver(_source(), config)
    return Audio8AsrInstance(config, TorchAudio8AsrEngine(config, resources))


AUDIO8_ASR_DEFINITION = ModelDefinition(
    manifest=AUDIO8_ASR_MANIFEST,
    runtime_factories={"pytorch-mps": _create_pytorch_mps},
    artifacts=AUDIO8_ASR_CONFIG.artifacts,
    resource_provider=Audio8AsrResourceProvider(_source()),
)


def register_audio8_asr(
    models: ModelRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned Audio8-ASR source model."""

    models.register(AUDIO8_ASR_DEFINITION, replace=replace)
    return AUDIO8_ASR_DEFINITION
