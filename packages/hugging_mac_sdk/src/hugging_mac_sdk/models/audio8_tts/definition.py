"""Audio8-TTS definition, runtime factory, and registration."""

from __future__ import annotations

from pathlib import Path

from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import (
    Audio8TtsCoreAIInstanceConfig,
    Audio8TtsInstanceConfig,
    Audio8TtsMlxInstanceConfig,
)
from .converter import Audio8TtsCoreAIConverter
from .coreai import CoreAIAudio8TtsEngine
from .instance import Audio8TtsInstance
from .mlx import MlxAudio8TtsEngine
from .mlx_resources import Audio8TtsMlxResourceResolver
from .resources import Audio8TtsCombinedResourceProvider, Audio8TtsResourceResolver
from .torch import TorchAudio8TtsEngine
from .utils.types import CoreAILayout

AUDIO8_TTS_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_MANIFEST = AUDIO8_TTS_CONFIG.manifest
COREAI_LAYOUT = CoreAILayout.model_validate(AUDIO8_TTS_CONFIG.extensions["audio8_tts_coreai"])
COREAI_CONVERTER = Audio8TtsCoreAIConverter(AUDIO8_TTS_CONFIG, COREAI_LAYOUT)


def _pytorch_artifact(variant: str, runtime: str = "pytorch") -> ModelArtifact:
    return AUDIO8_TTS_CONFIG.get_artifact("source", variant=variant, runtime=runtime)


def _pytorch_source(variant: str, runtime: str = "pytorch") -> HuggingFaceSource:
    source = _pytorch_artifact(variant, runtime).source
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


def _tokenizer_source(artifact: ModelArtifact) -> HuggingFaceSource:
    shared = _tokenizer_artifact(artifact)
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Audio8-TTS needs one shared tokenizer source")
    return shared.source


def _tokenizer_artifact(artifact: ModelArtifact) -> ModelArtifact:
    (share_id,) = artifact.required_shares
    return AUDIO8_TTS_CONFIG.get_artifact(share_id, shared=True)


def _create_pytorch(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsInstanceConfig.model_validate(options | {"runtime": "pytorch"})
    artifact = _pytorch_artifact(config.variant)
    tokenizer_artifact = _tokenizer_artifact(artifact)
    resources = Audio8TtsResourceResolver(
        _pytorch_source(config.variant),
        config,
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer_artifact,
        tokenizer_source=_tokenizer_source(artifact),
    )
    return Audio8TtsInstance(config, TorchAudio8TtsEngine(config, resources), AUDIO8_TTS_MANIFEST)


def _create_mlx(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    artifact = _mlx_artifact()
    tokenizer_artifact = _tokenizer_artifact(artifact)
    resources = Audio8TtsMlxResourceResolver(
        _mlx_source(),
        config,
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=artifact,
        tokenizer_artifact=tokenizer_artifact,
        tokenizer_source=_tokenizer_source(artifact),
    )
    return Audio8TtsInstance(config, MlxAudio8TtsEngine(config, resources), AUDIO8_TTS_MANIFEST)


def _create_coreai(options: dict[str, object]) -> Audio8TtsInstance:
    config = Audio8TtsCoreAIInstanceConfig.model_validate(options | {"runtime": "coreai"})
    artifact = AUDIO8_TTS_CONFIG.get_artifact(
        COREAI_LAYOUT.target_artifact, variant=config.variant, runtime="coreai"
    )
    source = _pytorch_artifact(config.variant)
    path = config.source_path or artifact.resolve(config.model_home)
    resources = Audio8TtsResourceResolver(
        _pytorch_source(config.variant),
        Audio8TtsInstanceConfig(
            variant=config.variant,
            model_home=config.model_home,
            source_path=path / COREAI_LAYOUT.pytorch_directory,
            tokenizer_path=config.tokenizer_path,
            hf_token=config.hf_token,
        ),
        manifest=AUDIO8_TTS_MANIFEST,
        artifact=source,
        tokenizer_artifact=_tokenizer_artifact(artifact),
        tokenizer_source=_tokenizer_source(artifact),
    )
    return Audio8TtsInstance(
        config,
        CoreAIAudio8TtsEngine(config, artifact, resources, COREAI_LAYOUT),
        AUDIO8_TTS_MANIFEST,
    )


AUDIO8_TTS_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_MANIFEST,
    runtime_factories={
        "pytorch": _create_pytorch,
        "mlx": _create_mlx,
        "coreai": _create_coreai,
    },
    artifacts=AUDIO8_TTS_CONFIG.artifacts,
    converter_ids=(COREAI_CONVERTER.converter_id,),
    resource_provider=Audio8TtsCombinedResourceProvider(
        AUDIO8_TTS_MANIFEST,
        {variant.name: _pytorch_artifact(variant.name) for variant in AUDIO8_TTS_MANIFEST.variants},
        {
            variant.name: _tokenizer_artifact(_pytorch_artifact(variant.name))
            for variant in AUDIO8_TTS_MANIFEST.variants
        },
        _mlx_source(),
        _mlx_artifact(),
        coreai_artifacts={
            a.variant: a
            for a in AUDIO8_TTS_CONFIG.artifacts
            if a.runtime == "coreai" and a.variant is not None
        },
        coreai_converter=COREAI_CONVERTER,
    ),
)


def register_audio8_tts(
    models: ModelRegistry, converters: ConverterRegistry | None = None, *, replace: bool = False
) -> ModelDefinition:
    """Register the Audio8-TTS Preview model."""

    if converters is not None:
        converters.register(COREAI_CONVERTER, replace=replace)
    models.register(AUDIO8_TTS_DEFINITION, replace=replace)
    return AUDIO8_TTS_DEFINITION
