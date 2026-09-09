"""Unlimited-OCR definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import UnlimitedOcrMlxInstanceConfig
from .instance import UnlimitedOcrInstance
from .mlx import MlxUnlimitedOcrEngine
from .resources import UnlimitedOcrResourceProvider, UnlimitedOcrResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
UNLIMITED_OCR_MANIFEST = _CONFIG.manifest
_ARTIFACT = _CONFIG.get_artifact("mlx-4bit", variant="4bit", runtime="mlx")
if not isinstance(_ARTIFACT.source, HuggingFaceSource):
    raise TypeError("Unlimited-OCR requires a Hugging Face snapshot source")
_SOURCE = _ARTIFACT.source


def _create(options: dict[str, object]) -> UnlimitedOcrInstance:
    config = UnlimitedOcrMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = UnlimitedOcrResourceResolver(
        _SOURCE, config, manifest=UNLIMITED_OCR_MANIFEST, artifact=_ARTIFACT
    )
    return UnlimitedOcrInstance(
        config, MlxUnlimitedOcrEngine(config, resources), UNLIMITED_OCR_MANIFEST
    )


UNLIMITED_OCR_DEFINITION = ModelDefinition(
    manifest=UNLIMITED_OCR_MANIFEST,
    runtime_factories={"mlx": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=UnlimitedOcrResourceProvider(_SOURCE, UNLIMITED_OCR_MANIFEST, _ARTIFACT),
)


def register_unlimited_ocr(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(UNLIMITED_OCR_DEFINITION, replace=replace)
    return UNLIMITED_OCR_DEFINITION
