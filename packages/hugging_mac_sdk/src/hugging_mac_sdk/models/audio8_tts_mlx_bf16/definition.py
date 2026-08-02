"""Audio8-TTS MLX BF16 definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Audio8TtsMlxBf16InstanceConfig
from .instance import Audio8TtsMlxBf16Instance
from .mlx import MlxAudio8TtsBf16Engine
from .resources import Audio8TtsMlxBf16ResourceProvider, Audio8TtsMlxBf16ResourceResolver

AUDIO8_TTS_MLX_BF16_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
AUDIO8_TTS_MLX_BF16_MANIFEST = AUDIO8_TTS_MLX_BF16_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = AUDIO8_TTS_MLX_BF16_MANIFEST.get_variant("bf16").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Audio8-TTS MLX BF16 needs one Hugging Face snapshot source")
    return resources[0]


def _create_mlx(options: dict[str, object]) -> Audio8TtsMlxBf16Instance:
    config = Audio8TtsMlxBf16InstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Audio8TtsMlxBf16ResourceResolver(_source(), config)
    return Audio8TtsMlxBf16Instance(config, MlxAudio8TtsBf16Engine(config, resources))


AUDIO8_TTS_MLX_BF16_DEFINITION = ModelDefinition(
    manifest=AUDIO8_TTS_MLX_BF16_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=AUDIO8_TTS_MLX_BF16_CONFIG.artifacts,
    resource_provider=Audio8TtsMlxBf16ResourceProvider(_source()),
)


def register_audio8_tts_mlx_bf16(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    models.register(AUDIO8_TTS_MLX_BF16_DEFINITION, replace=replace)
    return AUDIO8_TTS_MLX_BF16_DEFINITION
