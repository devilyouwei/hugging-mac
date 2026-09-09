"""GOT-OCR2.0 definition and registration."""

from pathlib import Path

from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import GotOcr2MlxInstanceConfig
from .instance import GotOcr2Instance
from .mlx import MlxGotOcr2Engine
from .resources import GotOcr2ResourceProvider, GotOcr2ResourceResolver

_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
GOT_OCR2_MANIFEST = _CONFIG.manifest
_ARTIFACT = _CONFIG.get_artifact("mlx-8bit", variant="8bit", runtime="mlx")
if not isinstance(_ARTIFACT.source, HuggingFaceSource):
    raise TypeError("GOT-OCR2.0 requires a Hugging Face snapshot source")
_SOURCE = _ARTIFACT.source


def _create(options: dict[str, object]) -> GotOcr2Instance:
    config = GotOcr2MlxInstanceConfig.model_validate(options | {"runtime": "mlx"})
    resources = GotOcr2ResourceResolver(
        _SOURCE, config, manifest=GOT_OCR2_MANIFEST, artifact=_ARTIFACT
    )
    return GotOcr2Instance(config, MlxGotOcr2Engine(config, resources), GOT_OCR2_MANIFEST)


GOT_OCR2_DEFINITION = ModelDefinition(
    manifest=GOT_OCR2_MANIFEST,
    runtime_factories={"mlx": _create},
    artifacts=_CONFIG.artifacts,
    resource_provider=GotOcr2ResourceProvider(_SOURCE, GOT_OCR2_MANIFEST, _ARTIFACT),
)


def register_got_ocr2(models: ModelRegistry, *, replace: bool = False) -> ModelDefinition:
    models.register(GOT_OCR2_DEFINITION, replace=replace)
    return GOT_OCR2_DEFINITION
