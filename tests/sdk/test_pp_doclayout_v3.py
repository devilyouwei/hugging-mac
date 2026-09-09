from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from hugging_mac_sdk.capabilities import DocumentLayoutAnalysis
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.pp_doclayout_v3 import (
    PP_DOCLAYOUT_V3_CONFIG,
    PP_DOCLAYOUT_V3_MANIFEST,
    register_pp_doclayout_v3,
)
from hugging_mac_sdk.models.pp_doclayout_v3.config import PPDocLayoutV3InstanceConfig
from hugging_mac_sdk.models.pp_doclayout_v3.coreml import CoreMlPPDocLayoutV3Engine
from hugging_mac_sdk.models.pp_doclayout_v3.instance import PPDocLayoutV3Instance
from hugging_mac_sdk.schemas.detection import ImageInput
from hugging_mac_sdk.schemas.document_layout import DocumentLayoutRequest
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, ResolvedResource
from PIL import Image


class FakeSession:
    device = "cpu-and-neural-engine"

    def __init__(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(inputs)
        return self.outputs

    async def close(self) -> None:
        return None


class FakeResources:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    async def resolve_coreml(self) -> ResolvedResource:
        return ResolvedResource(
            path=self.artifact,
            source=HuggingFaceSource(repo_id="PaddlePaddle/PP-DocLayoutV3_safetensors"),
            size_bytes=0,
        )


def fake_outputs() -> dict[str, Any]:
    logits = np.full((1, 3, 25), -20.0, dtype=np.float32)
    logits[0, 0, 22] = 8.0
    logits[0, 1, 6] = 9.0
    boxes = np.array(
        [[[0.5, 0.7, 0.8, 0.2], [0.5, 0.1, 0.8, 0.1], [0.5, 0.5, 0.1, 0.1]]],
        dtype=np.float32,
    )
    # Query 1 is before query 0. Exact values are less important than verifying
    # that confidence sorting does not become the public reading order.
    order_logits = np.zeros((1, 3, 3), dtype=np.float32)
    order_logits[0, 1, 0] = 10.0
    order_logits[0, 0, 1] = -10.0
    masks = np.full((1, 3, 200, 200), 10.0, dtype=np.float32)
    return {
        "logits": logits,
        "pred_boxes": boxes,
        "order_logits": order_logits,
        "out_masks": masks,
    }


async def test_coreml_layout_uses_fixed_preprocessing_and_returns_regions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    session = FakeSession(fake_outputs())

    async def fake_create_session(*_: Any, **__: Any) -> FakeSession:
        return session

    monkeypatch.setattr(
        "hugging_mac_sdk.models.pp_doclayout_v3.coreml.CoreMLProvider.create_session",
        fake_create_session,
    )
    artifact = tmp_path / "coreml-fp16"
    (artifact / "model.mlpackage").mkdir(parents=True)
    image = tmp_path / "page.png"
    Image.new("RGB", (1000, 500), "white").save(image)
    labels = tuple(
        PP_DOCLAYOUT_V3_CONFIG.extensions["pp_doclayout_v3"]["labels"]  # type: ignore[index]
    )
    config = PPDocLayoutV3InstanceConfig()
    engine = CoreMlPPDocLayoutV3Engine(config, FakeResources(artifact))  # type: ignore[arg-type]
    instance = PPDocLayoutV3Instance(
        config, engine, PP_DOCLAYOUT_V3_MANIFEST, input_size=800, labels=labels
    )

    await instance.load()
    response = await instance.require(DocumentLayoutAnalysis).analyze_layout(
        DocumentLayoutRequest(image=ImageInput(path=image), confidence=0.5)
    )

    assert [region.label for region in response.regions] == ["doc_title", "text"]
    assert [region.order for region in response.regions] == [0, 1]
    assert response.image_size.width == 1000
    assert response.image_size.height == 500
    assert session.calls[0]["pixel_values"].shape == (1, 3, 800, 800)
    assert session.calls[0]["pixel_values"].dtype == np.float32
    assert len(response.regions[0].polygon) >= 4


def test_manifest_registration_and_conversion_contract() -> None:
    assert PP_DOCLAYOUT_V3_MANIFEST.model_id == "paddlepaddle/pp-doclayout-v3"
    assert PP_DOCLAYOUT_V3_MANIFEST.capabilities == frozenset({"document-layout-analysis"})
    assert PP_DOCLAYOUT_V3_MANIFEST.default_runtime == "coreml"
    assert PP_DOCLAYOUT_V3_CONFIG.get_artifact(artifact_id="source").format == "safetensors"
    models = ModelRegistry()
    converters = ConverterRegistry()
    definition = register_pp_doclayout_v3(models, converters)
    for runtime in ("pytorch-mps", "coreml"):
        instance = definition.create(runtime=runtime, variant="base")
        assert isinstance(instance, PPDocLayoutV3Instance)
        assert instance.supports(DocumentLayoutAnalysis)
    assert converters.get("paddlepaddle.pp-doclayout-v3").converter_id == (
        "paddlepaddle.pp-doclayout-v3"
    )
