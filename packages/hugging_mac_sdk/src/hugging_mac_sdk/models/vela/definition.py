"""Vela definition, runtime factories, and registration."""

from pathlib import Path

from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.decision import DecisionSupport

from .config import VelaInstanceConfig
from .converter import VelaCoreAIConverter
from .coreai import CoreAIVelaEngine
from .instance import VelaInstance
from .resources import VelaResourceProvider, VelaResourceResolver
from .torch import TorchVelaEngine
from .utils.types import CoreAILayout

VELA_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
DECISION_SUPPORT = DecisionSupport.model_validate(VELA_CONFIG.extensions["decision"])
VELA_MANIFEST = VELA_CONFIG.manifest
COREAI_LAYOUT = CoreAILayout.model_validate(VELA_CONFIG.extensions["vela_coreai"])
VELA_CONVERTER = VelaCoreAIConverter(VELA_CONFIG, COREAI_LAYOUT)


def _create_coreai(options: dict[str, object]) -> VelaInstance:
    config = VelaInstanceConfig.model_validate(options | {"runtime": "coreai"})
    variant = VELA_MANIFEST.get_variant(config.variant)
    model_name = variant.metadata["response_model"]
    if not isinstance(model_name, str):
        raise TypeError("Vela response_model must be a string")
    resources = VelaResourceResolver(
        config,
        manifest=VELA_MANIFEST,
        artifact=VELA_CONFIG.get_artifact(
            shared=False, variant=config.variant, runtime=config.runtime
        ),
        shared_artifacts=VELA_DEFINITION.shared_artifacts,
    )
    return VelaInstance(
        config,
        CoreAIVelaEngine(config, resources, COREAI_LAYOUT, model_name=model_name),
        VELA_MANIFEST,
        DECISION_SUPPORT,
    )


def _create_pytorch(options: dict[str, object]) -> VelaInstance:
    config = VelaInstanceConfig.model_validate(options | {"runtime": "pytorch"})
    variant = VELA_MANIFEST.get_variant(config.variant)
    model_name = variant.metadata["response_model"]
    if not isinstance(model_name, str):
        raise TypeError("Vela response_model must be a string")
    resources = VelaResourceResolver(
        config,
        manifest=VELA_MANIFEST,
        artifact=VELA_CONFIG.get_artifact(shared=False, variant=config.variant, runtime="pytorch"),
        shared_artifacts=VELA_DEFINITION.shared_artifacts,
    )
    return VelaInstance(
        config,
        TorchVelaEngine(config, resources, model_name=model_name),
        VELA_MANIFEST,
        DECISION_SUPPORT,
    )


VELA_DEFINITION = ModelDefinition(
    manifest=VELA_MANIFEST,
    runtime_factories={"pytorch": _create_pytorch, "coreai": _create_coreai},
    artifacts=VELA_CONFIG.artifacts,
    converter_ids=(VELA_CONVERTER.converter_id,),
    resource_provider=VelaResourceProvider(VELA_CONFIG, VELA_CONVERTER),
)


def register_vela(
    models: ModelRegistry, converters: ConverterRegistry | None = None, *, replace: bool = False
) -> ModelDefinition:
    if converters is not None:
        converters.register(VELA_CONVERTER, replace=replace)
    models.register(VELA_DEFINITION, replace=replace)
    return VELA_DEFINITION
