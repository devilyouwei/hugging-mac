"""Audio8-TTS definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8TtsInstanceConfig, Audio8TtsMlxInstanceConfig
from .instance import Audio8TtsInstance
from .mlx import MlxAudio8TtsEngine
from .mlx_resources import Audio8TtsMlxResourceResolver
from .resources import Audio8TtsCombinedResourceProvider, Audio8TtsResourceResolver
from .torch import TorchAudio8TtsEngine

AUDIO8_TTS_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_MANIFEST = AUDIO8_TTS_CONFIG.manifest


def _sources() -> tuple[HuggingFaceSource, HuggingFaceSource]:
    resources = AUDIO8_TTS_MANIFEST.get_variant("0.6b-preview").resources
    if len(resources) != 2 or not all(isinstance(item, HuggingFaceSource) for item in resources):
        raise TypeError("Audio8-TTS needs PyTorch and MLX Hugging Face sources")
    return resources[0], resources[1]  # type: ignore[return-value]


def _tokenizer_source() -> HuggingFaceSource:
    shared = AUDIO8_TTS_CONFIG.artifacts[-1]
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Audio8-TTS needs one shared tokenizer source")
    return shared.source


def _create_pytorch(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsInstanceConfig.model_validate(
        options | {"runtime": "pytorch"}
    )
    resources = Audio8TtsResourceResolver(
        _sources()[0], config, tokenizer_source=_tokenizer_source()
    )
    return Audio8TtsInstance(config, TorchAudio8TtsEngine(config, resources))


def _create_mlx(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Audio8TtsMlxResourceResolver(
        _sources()[1], config, tokenizer_source=_tokenizer_source()
    )
    return Audio8TtsInstance(config, MlxAudio8TtsEngine(config, resources))


AUDIO8_TTS_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_MANIFEST,
    runtime_factories={"pytorch": _create_pytorch, "mlx": _create_mlx},
    artifacts=AUDIO8_TTS_CONFIG.artifacts,
    resource_provider=Audio8TtsCombinedResourceProvider(*_sources(), _tokenizer_source()),
)


def register_audio8_tts(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    """Register the pinned Audio8-TTS Preview model."""

    models.register(AUDIO8_TTS_DEFINITION, replace=replace)
    return AUDIO8_TTS_DEFINITION
