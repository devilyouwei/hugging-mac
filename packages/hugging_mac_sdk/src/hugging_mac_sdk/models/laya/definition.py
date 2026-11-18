"""Laya declarative registration and lazy runtime factories."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.decision import DecisionSupport

from .config import LayaInstanceConfig
from .coreai import CoreAILayaEngine
from .instance import LayaInstance
from .resources import LayaResourceProvider, LayaResourceResolver
from .torch import TorchLayaEngine

LAYA_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
LAYA_MANIFEST = LAYA_CONFIG.manifest
SUPPORT = DecisionSupport.model_validate(LAYA_CONFIG.extensions["decision"])


def _create(options: dict[str, object], runtime: str) -> LayaInstance:
    config = LayaInstanceConfig.model_validate(options | {"runtime": runtime})
    artifact = LAYA_CONFIG.get_artifact(shared=False, variant=config.variant, runtime=runtime)
    if runtime == "coreai" and config.max_length is not None:
        window = LAYA_MANIFEST.get_variant(config.variant).metadata["max_sequence_length"]
        if config.max_length != window:
            raise ValueError("The published Core AI window cannot be changed")
    resources = LayaResourceResolver(
        config,
        manifest=LAYA_MANIFEST,
        artifact=artifact,
        shared_artifacts=tuple(a for a in LAYA_CONFIG.artifacts if a.shared),
    )
    engine = (
        CoreAILayaEngine(
            config,
            resources,
            model_name=str(LAYA_MANIFEST.get_variant(config.variant).metadata["response_model"]),
        )
        if runtime == "coreai"
        else TorchLayaEngine(config, resources)
    )
    return LayaInstance(config, engine, LAYA_MANIFEST, SUPPORT)


def _create_coreai(options: dict[str, object]) -> LayaInstance:
    return _create(options, "coreai")


def _create_pytorch(options: dict[str, object]) -> LayaInstance:
    return _create({"variant": "multilingual"} | options, "pytorch")


LAYA_DEFINITION = ModelDefinition(
    manifest=LAYA_MANIFEST,
    runtime_factories={"coreai": _create_coreai, "pytorch": _create_pytorch},
    artifacts=LAYA_CONFIG.artifacts,
    resource_provider=LayaResourceProvider(LAYA_CONFIG),
)


def register_laya(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(LAYA_DEFINITION, replace=replace)
    return LAYA_DEFINITION
