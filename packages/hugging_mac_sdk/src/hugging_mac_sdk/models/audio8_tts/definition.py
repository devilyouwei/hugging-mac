"""Audio8-TTS definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8TtsInstanceConfig, Audio8TtsMlxInstanceConfig
from .instance import Audio8TtsInstance
from .mlx import MlxAudio8TtsEngine
from .mlx_resources import Audio8TtsMlxResourceResolver
from .resources import Audio8TtsCombinedResourceProvider, Audio8TtsResourceResolver
from .torch import TorchAudio8TtsEngine

AUDIO8_TTS_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_MANIFEST = AUDIO8_TTS_CONFIG.manifest


def _pytorch_artifact(variant: str) -> ModelArtifact:
    return AUDIO8_TTS_CONFIG.get_artifact("source", variant=variant, runtime="pytorch")


def _pytorch_source(variant: str) -> HuggingFaceSource:
    source = _pytorch_artifact(variant).source
    if not isinstance(source, HuggingFaceSource):
        raise TypeError(f"Audio8-TTS variant {variant} needs a PyTorch Hugging Face source")
    return source


def _mlx_source() -> HuggingFaceSource:
    source = _mlx_artifact().source
    if not isinstance(source, HuggingFaceSource):
        raise TypeError("Audio8-TTS needs one MLX Hugging Face source")
    return source


def _mlx_artifact() -> ModelArtifact:
    return AUDIO8_TTS_CONFIG.get_artifact("mlx-bf16", variant="0.6b-preview", runtime="mlx")


def _tokenizer_source() -> HuggingFaceSource:
    shared = _tokenizer_artifact()
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Audio8-TTS needs one shared tokenizer source")
    return shared.source


def _tokenizer_artifact() -> ModelArtifact:
    return AUDIO8_TTS_CONFIG.get_artifact("tokenizer", shared=True)


def _create_pytorch(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsInstanceConfig.model_validate(options | {"runtime": "pytorch"})
    resources = Audio8TtsResourceResolver(
        _pytorch_source(config.variant),
        config,
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=_pytorch_artifact(config.variant),
        tokenizer_artifact=_tokenizer_artifact(),
        tokenizer_source=_tokenizer_source(),
    )
    return Audio8TtsInstance(config, TorchAudio8TtsEngine(config, resources), AUDIO8_TTS_MANIFEST)


def _create_mlx(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Audio8TtsMlxResourceResolver(
        _mlx_source(),
        config,
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=_mlx_artifact(),
        tokenizer_artifact=_tokenizer_artifact(),
        tokenizer_source=_tokenizer_source(),
    )
    return Audio8TtsInstance(config, MlxAudio8TtsEngine(config, resources), AUDIO8_TTS_MANIFEST)


AUDIO8_TTS_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_MANIFEST,
    runtime_factories={"pytorch": _create_pytorch, "mlx": _create_mlx},
    artifacts=AUDIO8_TTS_CONFIG.artifacts,
    resource_provider=Audio8TtsCombinedResourceProvider(
        AUDIO8_TTS_MANIFEST,
        {variant.name: _pytorch_artifact(variant.name) for variant in AUDIO8_TTS_MANIFEST.variants},
        _tokenizer_source(),
        _tokenizer_artifact(),
        _mlx_source(),
        _mlx_artifact(),
    ),
)


def register_audio8_tts(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    """Register the Audio8-TTS Preview model."""

    models.register(AUDIO8_TTS_DEFINITION, replace=replace)
    return AUDIO8_TTS_DEFINITION
