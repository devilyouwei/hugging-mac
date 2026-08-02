"""Qwen3-TTS 0.6B Base 4-bit definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen3TtsInstanceConfig
from .instance import Qwen3TtsInstance
from .mlx import MlxQwen3TtsEngine
from .resources import Qwen3TtsResourceProvider, Qwen3TtsResourceResolver

QWEN3_TTS_0_6B_BASE_4BIT_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_TTS_0_6B_BASE_4BIT_MANIFEST = QWEN3_TTS_0_6B_BASE_4BIT_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = QWEN3_TTS_0_6B_BASE_4BIT_MANIFEST.get_variant("4bit").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Qwen3-TTS needs one Hugging Face snapshot source")
    return resources[0]


def _create_mlx(options: dict[str, object]) -> Qwen3TtsInstance:
    config = Qwen3TtsInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Qwen3TtsResourceResolver(_source(), config)
    return Qwen3TtsInstance(config, MlxQwen3TtsEngine(config, resources))


QWEN3_TTS_0_6B_BASE_4BIT_DEFINITION = ModelDefinition(
    manifest=QWEN3_TTS_0_6B_BASE_4BIT_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=QWEN3_TTS_0_6B_BASE_4BIT_CONFIG.artifacts,
    resource_provider=Qwen3TtsResourceProvider(_source()),
)


def register_qwen3_tts_0_6b_base_4bit(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    models.register(QWEN3_TTS_0_6B_BASE_4BIT_DEFINITION, replace=replace)
    return QWEN3_TTS_0_6B_BASE_4BIT_DEFINITION
