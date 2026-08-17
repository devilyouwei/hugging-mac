from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import FaceDetection
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.retinaface import RETINAFACE_MANIFEST, register_retinaface
from hugging_mac_sdk.models.retinaface.config import RetinaFaceInstanceConfig
from hugging_mac_sdk.models.retinaface.coreml import CoreMlRetinaFaceEngine
from hugging_mac_sdk.models.retinaface.instance import RetinaFaceInstance
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput
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
            source=HuggingFaceSource(repo_id="py-feat/retinaface"),
            size_bytes=0,
        )


def retinaface_outputs() -> dict[str, Any]:
    locations = np.zeros((1, 16_800, 4), dtype=np.float32)
    scores = np.zeros((1, 16_800, 2), dtype=np.float32)
    landmarks = np.zeros((1, 16_800, 10), dtype=np.float32)
    scores[..., 0] = 1.0
    scores[0, 100, 0] = 0.05
    scores[0, 100, 1] = 0.95
    return {"locations": locations, "scores": scores, "landmarks": landmarks}


async def test_coreml_retinaface_exposes_five_landmarks(tmp_path: Path, monkeypatch: Any) -> None:
    session = FakeSession(retinaface_outputs())

    async def fake_create_session(*_: Any, **__: Any) -> FakeSession:
        return session

    monkeypatch.setattr(
        "hugging_mac_sdk.models.retinaface.coreml.CoreMLProvider.create_session",
        fake_create_session,
    )
    artifact = tmp_path / "retinaface.mlpackage"
    artifact.mkdir()
    image = tmp_path / "face.jpg"
    Image.new("RGB", (320, 240)).save(image)
    config = RetinaFaceInstanceConfig(input_size=640)
    engine = CoreMlRetinaFaceEngine(config, FakeResources(artifact))  # type: ignore[arg-type]
    instance = RetinaFaceInstance(config, engine)

    await instance.load()
    response = await instance.require(FaceDetection).detect_faces(
        DetectionRequest(image=ImageInput(path=image), confidence=0.9)
    )

    assert instance.state is ModelState.READY
    assert response.runtime == "coreml"
    assert response.device == "cpu-and-neural-engine"
    assert response.image_size.width == 320
    assert response.image_size.height == 240
    assert len(response.faces) == 1
    assert response.faces[0].confidence == pytest.approx(0.95)
    assert response.faces[0].landmarks.left_eye.x >= 0
    assert response.faces[0].landmarks.right_mouth.y >= 0
    assert set(session.calls[0]) == {"image"}
    assert session.calls[0]["image"].shape == (1, 3, 640, 640)

    await instance.unload()
    assert instance.state is ModelState.UNLOADED


def test_retinaface_manifest_and_factories_match_capability() -> None:
    assert RETINAFACE_MANIFEST.model_id == "py-feat/retinaface"
    assert RETINAFACE_MANIFEST.capabilities == frozenset({"face-detection"})
    assert RETINAFACE_MANIFEST.default_runtime == "coreml"
    models = ModelRegistry()
    converters = ConverterRegistry()
    definition = register_retinaface(models, converters)
    for runtime in ("pytorch-mps", "coreml"):
        instance = definition.create(runtime=runtime, variant="mobilenet0.25")
        assert isinstance(instance, RetinaFaceInstance)
        assert instance.supports(FaceDetection)
        assert instance.info().runtime == runtime
    assert converters.get("py-feat.retinaface").converter_id == "py-feat.retinaface"
