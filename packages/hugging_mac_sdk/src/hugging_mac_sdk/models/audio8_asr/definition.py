"""Audio8-ASR definition, PyTorch factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8AsrInstanceConfig
from .converter import Audio8AsrConverter
from .coreml import CoreMlAudio8AsrEngine
from .instance import Audio8AsrInstance
from .resources import Audio8AsrResourceProvider, Audio8AsrResourceResolver
from .torch import TorchAudio8AsrEngine

AUDIO8_ASR_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_ASR_MANIFEST = AUDIO8_ASR_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = AUDIO8_ASR_MANIFEST.get_variant("0.1b").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Audio8-ASR needs one Hugging Face snapshot source")
    return resources[0]


def _tokenizer_source() -> HuggingFaceSource:
    shared = AUDIO8_ASR_CONFIG.artifacts[-1]
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Audio8-ASR needs one shared tokenizer source")
    return shared.source


def _create_pytorch_mps(options: dict[str, object]) -> Audio8AsrInstance:
    config = Audio8AsrInstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    resources = Audio8AsrResourceResolver(
        _source(), config, tokenizer_source=_tokenizer_source()
    )
    return Audio8AsrInstance(config, TorchAudio8AsrEngine(config, resources))


def _create_coreml(options: dict[str, object]) -> Audio8AsrInstance:
    normalized = dict(options)
    requested_device = normalized.pop("device", None)
    if requested_device is not None:
        normalized["compute_units"] = requested_device
    config = Audio8AsrInstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    resources = Audio8AsrResourceResolver(
        _source(), config, tokenizer_source=_tokenizer_source()
    )
    return Audio8AsrInstance(config, CoreMlAudio8AsrEngine(config, resources))


AUDIO8_ASR_DEFINITION = ModelDefinition(
    manifest=AUDIO8_ASR_MANIFEST,
    runtime_factories={
        "pytorch-mps": _create_pytorch_mps,
        "coreml": _create_coreml,
    },
    artifacts=AUDIO8_ASR_CONFIG.artifacts,
    converter_ids=("audio8.audio8-asr-0.1b",),
    resource_provider=Audio8AsrResourceProvider(_source(), _tokenizer_source()),
)


def register_audio8_asr(
    models: ModelRegistry,
    converters: ConverterRegistry | None = None,
    *,
    replace: bool = False,
) -> ModelDefinition:
    """Register the pinned Audio8-ASR source model."""

    if converters is not None:
        converters.register(Audio8AsrConverter(), replace=replace)
    models.register(AUDIO8_ASR_DEFINITION, replace=replace)
    return AUDIO8_ASR_DEFINITION
