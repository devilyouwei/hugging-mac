"""Qwen3.5 4B OptiQ 4-bit definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen35OptiQMlxInstanceConfig
from .instance import Qwen35OptiQMlxInstance
from .mlx import MlxQwen35OptiQEngine
from .resources import Qwen35OptiQMlxResourceProvider, Qwen35OptiQMlxResourceResolver

QWEN3_5_4B_OPTIQ_4BIT_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_5_4B_OPTIQ_4BIT_MANIFEST = QWEN3_5_4B_OPTIQ_4BIT_CONFIG.manifest


def _source() -> HuggingFaceSource:
    resources = QWEN3_5_4B_OPTIQ_4BIT_MANIFEST.get_variant("4bit").resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError("Qwen3.5 needs one Hugging Face snapshot source")
    return resources[0]


def _create_mlx(options: dict[str, object]) -> Qwen35OptiQMlxInstance:
    config = Qwen35OptiQMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = Qwen35OptiQMlxResourceResolver(_source(), config)
    return Qwen35OptiQMlxInstance(config, MlxQwen35OptiQEngine(config, resources))


QWEN3_5_4B_OPTIQ_4BIT_DEFINITION = ModelDefinition(
    manifest=QWEN3_5_4B_OPTIQ_4BIT_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=QWEN3_5_4B_OPTIQ_4BIT_CONFIG.artifacts,
    resource_provider=Qwen35OptiQMlxResourceProvider(_source()),
)


def register_qwen3_5_4b_optiq_4bit(
    models: ModelRegistry, *, replace: bool = False
) -> ModelDefinition:
    models.register(QWEN3_5_4B_OPTIQ_4BIT_DEFINITION, replace=replace)
    return QWEN3_5_4B_OPTIQ_4BIT_DEFINITION
