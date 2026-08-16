"""Qwen3.5 MLX definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Qwen35MlxInstanceConfig
from .instance import Qwen35MlxInstance
from .mlx import MlxQwen35Engine
from .resources import Qwen35MlxResourceProvider, Qwen35MlxResourceResolver

QWEN3_5_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
QWEN3_5_MANIFEST = QWEN3_5_CONFIG.manifest


def _source(variant: str) -> HuggingFaceSource:
    resources = QWEN3_5_MANIFEST.get_variant(variant).resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError(f"Qwen3.5 variant {variant} needs one Hugging Face snapshot source")
    return resources[0]


def _tokenizer_source() -> HuggingFaceSource:
    shared = QWEN3_5_CONFIG.artifacts[-1]
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Qwen3.5 needs one shared tokenizer source")
    return shared.source


def _create_mlx(options: dict[str, object]) -> Qwen35MlxInstance:
    config = Qwen35MlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    return Qwen35MlxInstance(
        config,
        MlxQwen35Engine(
            config,
            Qwen35MlxResourceResolver(
                _source(config.variant), config, tokenizer_source=_tokenizer_source()
            ),
        ),
    )


QWEN3_5_DEFINITION = ModelDefinition(
    manifest=QWEN3_5_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=QWEN3_5_CONFIG.artifacts,
    resource_provider=Qwen35MlxResourceProvider(
        {variant.name: _source(variant.name) for variant in QWEN3_5_MANIFEST.variants},
        _tokenizer_source(),
    ),
)


def register_qwen3_5(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(QWEN3_5_DEFINITION, replace=replace)
    return QWEN3_5_DEFINITION
