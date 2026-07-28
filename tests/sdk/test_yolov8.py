from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.yolov8 import (
    CoreMlYoloV8Instance,
    PyTorchMpsYoloV8Instance,
    register_yolov8,
)
from hugging_mac_sdk.models.yolov8.config import YoloV8InstanceConfig
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput
from hugging_mac_sdk.schemas.resources import ResolvedResource, UrlFileSource


class FakeTensor:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[Any]:
        return self._values


class FakeModel:
    def __init__(self, artifact: str) -> None:
        self.artifact = artifact
        self.calls: list[dict[str, object]] = []

    def predict(self, image: object, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10.0, 20.0, 110.0, 220.0]]),
            conf=FakeTensor([0.91]),
            cls=FakeTensor([0.0]),
        )
        return [
            SimpleNamespace(
                orig_shape=(480, 640),
                names={0: "person"},
                boxes=boxes,
                speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
            )
        ]


class FakeAssets:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    async def resolve_source(self) -> ResolvedResource:
        return self._resource()

    async def resolve_coreml(self) -> ResolvedResource:
        return self._resource()

    def _resource(self) -> ResolvedResource:
        size = 0 if self.artifact.is_dir() else self.artifact.stat().st_size
        return ResolvedResource(
            path=self.artifact,
            source=UrlFileSource(url="https://models.example/yolov8"),
            size_bytes=size,
        )


def install_fake_runtimes(monkeypatch: Any) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda artifact, **_: FakeModel(artifact)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
            mps=SimpleNamespace(empty_cache=lambda: None),
        ),
    )


async def test_pytorch_mps_instance_exposes_detection_capability(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    install_fake_runtimes(monkeypatch)
    artifact = tmp_path / "yolov8n.pt"
    artifact.write_bytes(b"weights")
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image bytes are not decoded for path inputs")
    instance = PyTorchMpsYoloV8Instance(
        YoloV8InstanceConfig(runtime="pytorch-mps"),
        FakeAssets(artifact),  # type: ignore[arg-type]
    )

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
    assert response.detections[0].confidence == 0.91
    assert response.detections[0].box.x2 == 110.0
    model = instance._model
    assert model.calls[0]["device"] == "mps"
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["classes"] == [0]

    await instance.unload()
    assert instance.state is ModelState.UNLOADED


async def test_coreml_instance_does_not_pass_torch_device(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    install_fake_runtimes(monkeypatch)
    artifact = tmp_path / "yolov8n.mlpackage"
    artifact.mkdir()
    image = tmp_path / "input.jpg"
    image.write_bytes(b"input")
    instance = CoreMlYoloV8Instance(
        YoloV8InstanceConfig(runtime="coreml"),
        FakeAssets(artifact),  # type: ignore[arg-type]
    )

    await instance.load()
    response = await instance.require(ObjectDetection).detect(
        DetectionRequest(image=ImageInput(path=image))
    )

    assert response.runtime == "coreml"
    assert response.device == "all"
    assert "device" not in instance._model.calls[0]


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

        assert isinstance(coreml, CoreMlYoloV8Instance)
        assert isinstance(pytorch, PyTorchMpsYoloV8Instance)
        assert coreml.info().variant == variant
        assert pytorch.info().variant == variant


def test_definition_factory_accepts_explicit_runtime() -> None:
    models = ModelRegistry()
    from hugging_mac_sdk.converters.registry import ConverterRegistry

    definition = register_yolov8(models, ConverterRegistry())

    coreml = definition.create(runtime="coreml")
    pytorch = definition.create(runtime="pytorch-mps")

    assert isinstance(coreml, CoreMlYoloV8Instance)
    assert isinstance(pytorch, PyTorchMpsYoloV8Instance)
