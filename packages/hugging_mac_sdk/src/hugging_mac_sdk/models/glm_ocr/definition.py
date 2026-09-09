"""GLM-OCR definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import GlmOcrMlxInstanceConfig
from .instance import GlmOcrInstance
from .mlx import MlxGlmOcrEngine
from .resources import GlmOcrResourceProvider, GlmOcrResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
GLM_OCR_MANIFEST = _CONFIG.manifest
_ARTIFACT = _CONFIG.get_artifact("mlx-8bit", variant="8bit", runtime="mlx")
if not isinstance(_ARTIFACT.source, HuggingFaceSource):
    raise TypeError("GLM-OCR requires a Hugging Face snapshot source")
_SOURCE = _ARTIFACT.source


def _create(options: dict[str, object]) -> GlmOcrInstance:
    config = GlmOcrMlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = GlmOcrResourceResolver(
        _SOURCE, config, manifest=GLM_OCR_MANIFEST, artifact=_ARTIFACT
    )
    return GlmOcrInstance(config, MlxGlmOcrEngine(config, resources), GLM_OCR_MANIFEST)


GLM_OCR_DEFINITION = ModelDefinition(
    manifest=GLM_OCR_MANIFEST,
    runtime_factories={"mlx": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=GlmOcrResourceProvider(_SOURCE, GLM_OCR_MANIFEST, _ARTIFACT),
)


def register_glm_ocr(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(GLM_OCR_DEFINITION, replace=replace)
    return GLM_OCR_DEFINITION
