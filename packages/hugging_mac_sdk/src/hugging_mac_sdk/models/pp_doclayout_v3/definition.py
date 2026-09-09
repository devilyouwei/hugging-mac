"""PP-DocLayoutV3 factories and SDK registration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from hugging_mac_sdk.converters.registry import ConverterRegistry
from hugging_mac_sdk.core.config import load_model_config
from hugging_mac_sdk.core.registry import ModelDefinition, ModelRegistry
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

from .config import PPDocLayoutV3InstanceConfig
from .converter import PPDocLayoutV3Converter
from .coreml import CoreMlPPDocLayoutV3Engine
from .instance import PPDocLayoutV3Instance
from .resources import PPDocLayoutV3ResourceProvider, PPDocLayoutV3ResourceResolver
from .torch import TorchPPDocLayoutV3Engine

PP_DOCLAYOUT_V3_CONFIG = load_model_config(Path(__file__).with_name("model.yaml"))
PP_DOCLAYOUT_V3_MANIFEST = PP_DOCLAYOUT_V3_CONFIG.manifest
PP_DOCLAYOUT_V3_CONVERTER = PPDocLayoutV3Converter(PP_DOCLAYOUT_V3_MANIFEST.model_id)


def _source() -> HuggingFaceSource:
    source = PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="source").source
    if not isinstance(source, HuggingFaceSource):
        raise TypeError("PP-DocLayoutV3 requires one Hugging Face safetensors source")
    return source


def _settings() -> tuple[int, tuple[str, ...]]:
    raw = PP_DOCLAYOUT_V3_CONFIG.extensions.get("pp_doclayout_v3")
    if not isinstance(raw, dict):
        raise TypeError("PP-DocLayoutV3 package extension is missing")
    labels = raw.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise TypeError("PP-DocLayoutV3 label map is invalid")
    input_size = raw.get("input_size", 800)
    if not isinstance(input_size, int):
        raise TypeError("PP-DocLayoutV3 input size is invalid")
    return input_size, tuple(cast(list[str], labels))


def _resources(config: PPDocLayoutV3InstanceConfig) -> PPDocLayoutV3ResourceResolver:
    return PPDocLayoutV3ResourceResolver(
        _source(),
        config,
        PP_DOCLAYOUT_V3_MANIFEST,
        PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="source"),
        PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="coreml-fp16"),
        converter=PP_DOCLAYOUT_V3_CONVERTER,
    )


def _create_pytorch(options: dict[str, object]) -> PPDocLayoutV3Instance:
    config = PPDocLayoutV3InstanceConfig.model_validate(options | {"runtime": "pytorch-mps"})
    input_size, labels = _settings()
    return PPDocLayoutV3Instance(
        config,
        TorchPPDocLayoutV3Engine(config, _resources(config)),
        PP_DOCLAYOUT_V3_MANIFEST,
        input_size=input_size,
        labels=labels,
    )


def _create_coreml(options: dict[str, object]) -> PPDocLayoutV3Instance:
    normalized = dict(options)
    device = normalized.pop("device", None)
    if device is not None:
        normalized["compute_units"] = device
    config = PPDocLayoutV3InstanceConfig.model_validate(normalized | {"runtime": "coreml"})
    input_size, labels = _settings()
    return PPDocLayoutV3Instance(
        config,
        CoreMlPPDocLayoutV3Engine(config, _resources(config)),
        PP_DOCLAYOUT_V3_MANIFEST,
        input_size=input_size,
        labels=labels,
    )


PP_DOCLAYOUT_V3_DEFINITION = ModelDefinition(
    manifest=PP_DOCLAYOUT_V3_MANIFEST,
    runtime_factories={"pytorch-mps": _create_pytorch, "coreml": _create_coreml},
    artifacts=PP_DOCLAYOUT_V3_CONFIG.artifacts,
    converter_ids=(PP_DOCLAYOUT_V3_CONVERTER.converter_id,),
    resource_provider=PPDocLayoutV3ResourceProvider(
        _source(),
        PP_DOCLAYOUT_V3_MANIFEST,
        PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="source"),
        PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="coreml-fp16"),
    ),
)


def register_pp_doclayout_v3(
    models: ModelRegistry,
    converters: ConverterRegistry,
    *,
    replace: bool = False,
) -> ModelDefinition:
    converters.register(PP_DOCLAYOUT_V3_CONVERTER, replace=replace)
    models.register(PP_DOCLAYOUT_V3_DEFINITION, replace=replace)
    return PP_DOCLAYOUT_V3_DEFINITION
