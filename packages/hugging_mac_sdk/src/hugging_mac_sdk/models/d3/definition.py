"""D3 definition, runtime factories, and registration."""

from pathlib import Path

from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.decision import DecisionSupport

from .config import D3InstanceConfig
from .converter import D3CoreAIConverter
from .coreai import CoreAID3Engine
from .instance import D3Instance
from .resources import D3ResourceProvider, D3ResourceResolver
from .torch import TorchD3Engine
from .utils.types import CoreAILayout

D3_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
DECISION_SUPPORT = DecisionSupport.model_validate(D3_CONFIG.extensions["decision"])
D3_MANIFEST = D3_CONFIG.manifest
COREAI_LAYOUT = CoreAILayout.model_validate(D3_CONFIG.extensions["d3_coreai"])
D3_CONVERTER = D3CoreAIConverter(D3_CONFIG, COREAI_LAYOUT)


def _create_coreai(options: dict[str, object]) -> D3Instance:
    config = D3InstanceConfig.model_validate(options | {"runtime": "coreai"})
    variant = D3_MANIFEST.get_variant(config.variant)
    model_name = variant.metadata["response_model"]
    if not isinstance(model_name, str):
        raise TypeError("D3 response_model must be a string")
    resources = D3ResourceResolver(
        config,
        manifest=D3_MANIFEST,
        artifact=D3_CONFIG.get_artifact(
            shared=False, variant=config.variant, runtime=config.runtime
        ),
        shared_artifacts=D3_DEFINITION.shared_artifacts,
    )
    return D3Instance(
        config,
        CoreAID3Engine(config, resources, COREAI_LAYOUT, model_name=model_name),
        D3_MANIFEST,
        DECISION_SUPPORT,
    )


def _create_pytorch(options: dict[str, object]) -> D3Instance:
    config = D3InstanceConfig.model_validate(options | {"runtime": "pytorch"})
    variant = D3_MANIFEST.get_variant(config.variant)
    model_name = variant.metadata["response_model"]
    if not isinstance(model_name, str):
        raise TypeError("D3 response_model must be a string")
    resources = D3ResourceResolver(
        config,
        manifest=D3_MANIFEST,
        artifact=D3_CONFIG.get_artifact(shared=False, variant=config.variant, runtime="pytorch"),
        shared_artifacts=D3_DEFINITION.shared_artifacts,
    )
    return D3Instance(
        config,
        TorchD3Engine(config, resources, model_name=model_name),
        D3_MANIFEST,
        DECISION_SUPPORT,
    )


D3_DEFINITION = ModelDefinition(
    manifest=D3_MANIFEST,
    runtime_factories={"pytorch": _create_pytorch, "coreai": _create_coreai},
    artifacts=D3_CONFIG.artifacts,
    converter_ids=(D3_CONVERTER.converter_id,),
    resource_provider=D3ResourceProvider(D3_CONFIG, D3_CONVERTER),
)


def register_d3(
    models: ModelRegistry, converters: ConverterRegistry | None = None, *, replace: bool = False
) -> ModelDefinition:
    if converters is not None:
        converters.register(D3_CONVERTER, replace=replace)
    models.register(D3_DEFINITION, replace=replace)
    return D3_DEFINITION
