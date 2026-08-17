"""Gemma 4 MLX definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import Gemma4MlxInstanceConfig
from .instance import Gemma4MlxInstance
from .mlx import MlxGemma4Engine
from .resources import Gemma4MlxResourceProvider, Gemma4MlxResourceResolver

GEMMA_4_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
GEMMA_4_MANIFEST = GEMMA_4_CONFIG.manifest


def _source(variant: str) -> HuggingFaceSource:
    resources = GEMMA_4_MANIFEST.get_variant(variant).resources
    if len(resources) != 1 or not isinstance(resources[0], HuggingFaceSource):
        raise TypeError(f"Gemma 4 variant {variant} needs one Hugging Face snapshot source")
    return resources[0]


def _tokenizer_source() -> HuggingFaceSource:
    shared = GEMMA_4_CONFIG.artifacts[-1]
    if not shared.shared or not isinstance(shared.source, HuggingFaceSource):
        raise TypeError("Gemma 4 needs one shared tokenizer source")
    return shared.source


def _create_mlx(options: dict[str, object]) -> Gemma4MlxInstance:
    config = Gemma4MlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resolver = Gemma4MlxResourceResolver(
        _source(config.variant), config, tokenizer_source=_tokenizer_source()
    )
    return Gemma4MlxInstance(config, MlxGemma4Engine(config, resolver))


GEMMA_4_DEFINITION = ModelDefinition(
    manifest=GEMMA_4_MANIFEST,
    runtime_factories={"mlx": _create_mlx},
    artifacts=GEMMA_4_CONFIG.artifacts,
    resource_provider=Gemma4MlxResourceProvider(
        {variant.name: _source(variant.name) for variant in GEMMA_4_MANIFEST.variants},
        _tokenizer_source(),
    ),
)


def register_gemma_4(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(GEMMA_4_DEFINITION, replace=replace)
    return GEMMA_4_DEFINITION
