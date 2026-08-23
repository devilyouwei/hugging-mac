from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.yolov8 import YOLOV8_MANIFEST, register_yolov8
from hugging_mac_sdk.models.yolov8.config import YoloV8InstanceConfig
from hugging_mac_sdk.models.yolov8.coreml import CoreMlYoloV8Engine
from hugging_mac_sdk.models.yolov8.instance import YoloV8Instance
from hugging_mac_sdk.models.yolov8.torch import TorchYoloV8Engine
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput
from hugging_mac_sdk.schemas.resources import ResolvedResource, UrlFileSource
from PIL import Image


class FakeSession:
    def __init__(self, outputs: dict[str, Any], device: str) -> None:
        self.outputs = outputs
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(inputs)
        return self.outputs

    async def close(self) -> None:
        return None


class FakeAssets:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    async def resolve_source(self) -> ResolvedResource:
        return self._resource()

    async def resolve_coreml(self) -> ResolvedResource:
        return self._resource()

    async def resolve_onnx(self) -> ResolvedResource:
        return self._resource()

    def _resource(self) -> ResolvedResource:
        size = 0 if self.artifact.is_dir() else self.artifact.stat().st_size
        return ResolvedResource(
            path=self.artifact,
            source=UrlFileSource(url="https://models.example/yolov8"),
            size_bytes=size,
        )


def detection_outputs() -> dict[str, Any]:
    prediction = np.zeros((1, 84, 8400), dtype=np.float32)
    prediction[0, :4, 0] = (60, 200, 100, 200)
    prediction[0, 4, 0] = 0.91
    return {"predictions": prediction}


async def test_pytorch_mps_engine_exposes_detection_capability(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session = FakeSession(detection_outputs(), "mps")

    async def fake_create_session(*_: Any, **__: Any) -> FakeSession:
        return session

    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8.torch.TorchProvider.create_session",
        fake_create_session,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8.torch.torch_input",
        lambda *_: object(),
    )
    artifact = tmp_path / "yolov8n.pt"
    artifact.write_bytes(b"weights")
    image = tmp_path / "input.jpg"
    Image.new("RGB", (640, 480)).save(image)
    config = YoloV8InstanceConfig(runtime="pytorch-mps")
    engine = TorchYoloV8Engine(config, FakeAssets(artifact))  # type: ignore[arg-type]
    instance = YoloV8Instance(config, engine, YOLOV8_MANIFEST)

    await instance.load()
    detector = instance.require(ObjectDetection)
    response = await detector.detect(
        DetectionRequest(
            image=ImageInput(path=image),
            confidence=0.4,
            classes=(0,),
        )
    )

    assert instance.state is ModelState.READY
    assert response.runtime == "pytorch-mps"
    assert response.device == "mps"
    assert response.image_size.width == 640
    assert response.image_size.height == 480
    assert len(response.detections) == 1
    assert response.detections[0].label == "person"
    assert response.detections[0].confidence == pytest.approx(0.91)
    assert response.detections[0].box.x2 == 110.0
    assert set(session.calls[0]) == {"input"}

    await instance.unload()
    assert instance.state is ModelState.UNLOADED


async def test_coreml_engine_does_not_pass_torch_device(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session = FakeSession(detection_outputs(), "all")

    async def fake_create_session(*_: Any, **__: Any) -> FakeSession:
        return session

    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8.coreml.CoreMLProvider.create_session",
        fake_create_session,
    )
    artifact = tmp_path / "yolov8n.mlpackage"
    artifact.mkdir()
    image = tmp_path / "input.jpg"
    Image.new("RGB", (640, 480)).save(image)
    config = YoloV8InstanceConfig(runtime="coreml")
    engine = CoreMlYoloV8Engine(config, FakeAssets(artifact))  # type: ignore[arg-type]
    instance = YoloV8Instance(config, engine, YOLOV8_MANIFEST)

    await instance.load()
    response = await instance.require(ObjectDetection).detect(
        DetectionRequest(image=ImageInput(path=image))
    )

    assert response.runtime == "coreml"
    assert response.device == "all"
    assert "image" in session.calls[0]
    assert "device" not in session.calls[0]


def test_registered_factory_selects_runtime() -> None:
    models = ModelRegistry()
    from hugging_mac_sdk.converters.registry import ConverterRegistry

    register_yolov8(models, ConverterRegistry())

    for variant in ("n", "s", "m"):
        coreml = models.create_instance(
            "ultralytics/yolov8",
            variant=variant,
            runtime="coreml",
        )
        pytorch = models.create_instance(
            "ultralytics/yolov8",
            variant=variant,
            runtime="pytorch-mps",
        )
        onnx = models.create_instance(
            "ultralytics/yolov8",
            variant=variant,
            runtime="onnx",
        )

        assert isinstance(coreml, YoloV8Instance)
        assert isinstance(pytorch, YoloV8Instance)
        assert isinstance(onnx, YoloV8Instance)
        assert coreml.info().runtime == "coreml"
        assert pytorch.info().runtime == "pytorch-mps"
        assert onnx.info().runtime == "onnx"
        assert coreml.info().variant == variant
        assert pytorch.info().variant == variant


def test_definition_factory_accepts_explicit_runtime() -> None:
    models = ModelRegistry()
    from hugging_mac_sdk.converters.registry import ConverterRegistry

    definition = register_yolov8(models, ConverterRegistry())

    coreml = definition.create(runtime="coreml")
    pytorch = definition.create(runtime="pytorch-mps")
    onnx = definition.create(runtime="onnx")

    assert isinstance(coreml, YoloV8Instance)
    assert isinstance(pytorch, YoloV8Instance)
    assert isinstance(onnx, YoloV8Instance)
    assert coreml.info().runtime == "coreml"
    assert pytorch.info().runtime == "pytorch-mps"
    assert onnx.info().runtime == "onnx"
