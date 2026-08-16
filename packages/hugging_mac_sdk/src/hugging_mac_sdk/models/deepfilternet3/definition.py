"""DeepFilterNet3 definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import DeepFilterNet3InstanceConfig
from .coreml import CoreMlDeepFilterNet3Engine
from .instance import DeepFilterNet3Instance
from .resources import DeepFilterNet3ResourceProvider, DeepFilterNet3ResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
DEEPFILTERNET3_MANIFEST = _CONFIG.manifest
_SOURCE_RESOURCE = DEEPFILTERNET3_MANIFEST.get_variant("default").resources[0]
if not isinstance(_SOURCE_RESOURCE, HuggingFaceSource):
    raise TypeError("DeepFilterNet3 requires a Hugging Face source")
_SOURCE: HuggingFaceSource = _SOURCE_RESOURCE


def _create(options: dict[str, object]) -> DeepFilterNet3Instance:
    config = DeepFilterNet3InstanceConfig.model_validate(options | {"runtime": "coreml"})
    resources = DeepFilterNet3ResourceResolver(_SOURCE, config)
    return DeepFilterNet3Instance(config, CoreMlDeepFilterNet3Engine(config, resources))


DEEPFILTERNET3_DEFINITION = ModelDefinition(
    manifest=DEEPFILTERNET3_MANIFEST,
    runtime_factories={"coreml": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=DeepFilterNet3ResourceProvider(_SOURCE),
)


def register_deepfilternet3(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(DEEPFILTERNET3_DEFINITION, replace=replace)
    return DEEPFILTERNET3_DEFINITION
