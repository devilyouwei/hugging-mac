"""MOSS-TTS-Nano definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import MossTtsNanoInstanceConfig
from .instance import MossTtsNanoInstance
from .mlx import MlxMossTtsNanoEngine
from .resources import MossTtsNanoResourceProvider, MossTtsNanoResourceResolver

MOSS_TTS_NANO_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
MOSS_TTS_NANO_MANIFEST = MOSS_TTS_NANO_CONFIG.manifest


def _artifact() -> ModelArtifact:
    return MOSS_TTS_NANO_CONFIG.get_artifact("mlx-fp16", variant="nano-100m", runtime="mlx")


def _audio_tokenizer_artifact() -> ModelArtifact:
    return MOSS_TTS_NANO_CONFIG.get_artifact(artifact_id="audio-tokenizer")


def _source(artifact: ModelArtifact, label: str) -> HuggingFaceSource:
    if not isinstance(artifact.source, HuggingFaceSource):
        raise TypeError(f"MOSS-TTS-Nano needs a Hugging Face {label} source")
    return artifact.source


def _create_mlx(options: dict[str, object]) -> MossTtsNanoInstance:
    config = MossTtsNanoInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = MossTtsNanoResourceResolver(
        _source(_artifact(), "model"),
        _source(_audio_tokenizer_artifact(), "audio tokenizer"),
        config,
        MOSS_TTS_NANO_MANIFEST,
        _artifact(),
        _audio_tokenizer_artifact(),
    )
    return MossTtsNanoInstance(
        config, MlxMossTtsNanoEngine(config, resources), MOSS_TTS_NANO_MANIFEST
    )


MOSS_TTS_NANO_DEFINITION = ModelDefinition(
    manifest=MOSS_TTS_NANO_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=MOSS_TTS_NANO_CONFIG.artifacts,
    resource_provider=MossTtsNanoResourceProvider(
        _source(_artifact(), "model"),
        _source(_audio_tokenizer_artifact(), "audio tokenizer"),
        MOSS_TTS_NANO_MANIFEST,
        _artifact(),
        _audio_tokenizer_artifact(),
    ),
)


def register_moss_tts_nano(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(MOSS_TTS_NANO_DEFINITION, replace=replace)
    return MOSS_TTS_NANO_DEFINITION
