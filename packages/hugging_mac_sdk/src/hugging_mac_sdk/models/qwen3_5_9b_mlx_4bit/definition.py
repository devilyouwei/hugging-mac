"""Qwen3.5 9B MLX 4-bit definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen35MlxInstanceConfig
from .instance import Qwen35MlxInstance
from .mlx import MlxQwen35Engine
from .resources import Qwen35MlxResourceProvider, Qwen35MlxResourceResolver

QWEN3_5_9B_MLX_4BIT_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_5_9B_MLX_4BIT_MANIFEST = QWEN3_5_9B_MLX_4BIT_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = QWEN3_5_9B_MLX_4BIT_MANIFEST.get_variant("4bit").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Qwen3.5 needs one Hugging Face snapshot source")
    return resources[0]


def _create_mlx(options: dict[str, object]) -> Qwen35MlxInstance:
    config = Qwen35MlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Qwen35MlxResourceResolver(_source(), config)
    return Qwen35MlxInstance(config, MlxQwen35Engine(config, resources))


QWEN3_5_9B_MLX_4BIT_DEFINITION = ModelDefinition(
    manifest=QWEN3_5_9B_MLX_4BIT_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=QWEN3_5_9B_MLX_4BIT_CONFIG.artifacts,
    resource_provider=Qwen35MlxResourceProvider(_source()),
)


def register_qwen3_5_9b_mlx_4bit(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    models.register(QWEN3_5_9B_MLX_4BIT_DEFINITION, replace=replace)
    return QWEN3_5_9B_MLX_4BIT_DEFINITION
