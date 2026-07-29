from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import InstanceSegmentation, PoseEstimation
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.models.yolov8_pose import (
    YOLOV8_POSE_MANIFEST,
    CoreMlYoloV8PoseInstance,
    PyTorchMpsYoloV8PoseInstance,
    register_yolov8_pose,
)
from hugging_mac_sdk.models.yolov8_pose.config import YoloV8PoseInstanceConfig
from hugging_mac_sdk.models.yolov8_pose.converter import YoloV8PoseConverter
from hugging_mac_sdk.models.yolov8_seg import (
    YOLOV8_SEG_MANIFEST,
    CoreMlYoloV8SegInstance,
    PyTorchMpsYoloV8SegInstance,
    register_yolov8_seg,
)
from hugging_mac_sdk.models.yolov8_seg.config import YoloV8SegInstanceConfig
from hugging_mac_sdk.models.yolov8_seg.converter import YoloV8SegConverter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput
from hugging_mac_sdk.schemas.resources import ResolvedResource, UrlFileSource


class FakeTensor:
    def __init__(self, value: Any) -> None:
        self._value = value

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> Any:
        return self._value


class FakeNumpyContour:
    """Minimal NumPy-like contour that rejects ambiguous truth-value checks."""

    def __init__(self, points: list[list[float]]) -> None:
        self._points = points

    def __iter__(self) -> Any:
        return iter(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array is ambiguous")


class FakeAssets:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def resolve_source(self) -> ResolvedResource:
        return self._resource()

    async def resolve_coreml(self) -> ResolvedResource:
        return self._resource()

    def _resource(self) -> ResolvedResource:
        return ResolvedResource(
            path=self._path,
            source=UrlFileSource(url="https://models.example/yolo.pt"),
            size_bytes=0,
        )


class FakePoseModel:
    def predict(self, *_: object, **__: object) -> list[object]:
        return [
            SimpleNamespace(
                orig_shape=(480, 640),
                names={0: "person"},
                boxes=SimpleNamespace(
                    xyxy=FakeTensor([[10, 20, 110, 220]]),
                    conf=FakeTensor([0.9]),
                    cls=FakeTensor([0]),
                ),
                keypoints=SimpleNamespace(
                    xy=FakeTensor([[[15, 25], [20, 30]]]),
                    conf=FakeTensor([[0.8, 0.7]]),
                ),
                speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
            )
        ]


class FakeSegModel:
    def predict(self, *_: object, **__: object) -> list[object]:
        return [
            SimpleNamespace(
                orig_shape=(480, 640),
                names={0: "person"},
                boxes=SimpleNamespace(
                    xyxy=FakeTensor([[10, 20, 110, 220]]),
                    conf=FakeTensor([0.9]),
                    cls=FakeTensor([0]),
                ),
                masks=SimpleNamespace(
                    xy=[FakeNumpyContour([[10, 20], [110, 20], [110, 220]])]
                ),
                speed={},
            )
        ]


def test_pose_and_seg_manifests_register_all_supported_variants() -> None:
    sdk = ModelSdk()
    converters = ConverterRegistry()
    pose = register_yolov8_pose(sdk.registry, converters)
    seg = register_yolov8_seg(sdk.registry, converters)

    assert tuple(item.name for item in pose.manifest.variants) == ("n", "s", "m")
    assert tuple(item.name for item in seg.manifest.variants) == ("n", "s", "m")
    pose_source = YOLOV8_POSE_MANIFEST.get_variant("s").resources[0]
    seg_source = YOLOV8_SEG_MANIFEST.get_variant("m").resources[0]
    assert str(pose_source.url).endswith("yolov8s-pose.pt")  # type: ignore[union-attr]
    assert str(seg_source.url).endswith("yolov8m-seg.pt")  # type: ignore[union-attr]

    for variant in ("n", "s", "m"):
        assert isinstance(
            sdk.registry.create_instance(
                "ultralytics/yolov8-pose",
                variant=variant,
                runtime="coreml",
            ),
            CoreMlYoloV8PoseInstance,
        )
        assert isinstance(
            sdk.registry.create_instance(
                "ultralytics/yolov8-seg",
                variant=variant,
                runtime="pytorch-mps",
            ),
            PyTorchMpsYoloV8SegInstance,
        )


async def test_pose_and_seg_instances_map_task_specific_results(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    pose_weights = tmp_path / "pose.pt"
    pose_weights.write_bytes(b"weights")
    seg_package = tmp_path / "seg.mlpackage"
    seg_package.mkdir()
    models = iter((FakePoseModel(), FakeSegModel()))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda *_args, **_kwargs: next(models)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            mps=SimpleNamespace(empty_cache=lambda: None),
        ),
    )

    pose = PyTorchMpsYoloV8PoseInstance(
        YoloV8PoseInstanceConfig(runtime="pytorch-mps", variant="s"),
        FakeAssets(pose_weights),  # type: ignore[arg-type]
    )
    seg = CoreMlYoloV8SegInstance(
        YoloV8SegInstanceConfig(runtime="coreml", variant="m"),
        FakeAssets(seg_package),  # type: ignore[arg-type]
    )
    request = DetectionRequest(image=ImageInput(path=image))

    await pose.load()
    pose_response = await pose.require(PoseEstimation).estimate_pose(request)  # type: ignore[type-abstract]
    await seg.load()
    seg_response = await seg.require(InstanceSegmentation).segment(request)  # type: ignore[type-abstract]

    assert pose_response.poses[0].keypoints[1].confidence == 0.7
    assert pose_response.poses[0].label == "person"
    assert seg_response.segments[0].polygons[0][2].y == 220
    assert seg_response.segments[0].label == "person"


async def test_pose_and_seg_resources_use_independent_variant_directories(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_download(
        _: ResourceDownloader,
        source: Any,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")
        return ResolvedResource(path=destination, source=source, digest="1" * 64, size_bytes=7)

    async def fake_convert(
        _: YoloV8PoseConverter,
        request: ConversionRequest,
    ) -> ConversionResult:
        request.output_path.mkdir(parents=True, exist_ok=True)
        (request.output_path / "model.mlmodel").write_bytes(b"coreml")
        return ConversionResult(
            path=request.output_path,
            format=request.target_format,
            digest="2" * 64,
            size_bytes=6,
            converter_id="ultralytics.yolov8-pose",
            model_id=request.model_id,
            model_revision=request.model_revision,
            variant=request.variant,
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    monkeypatch.setattr(YoloV8PoseConverter, "convert", fake_convert)
    sdk = ModelSdk()
    converters = ConverterRegistry()
    register_yolov8_pose(sdk.registry, converters)
    register_yolov8_seg(sdk.registry, converters)
    options = {"model_home": tmp_path}

    pose = await sdk.resources.download_source(
        "ultralytics/yolov8-pose",
        variant="s",
        options=options,
    )
    seg = await sdk.resources.download_source(
        "ultralytics/yolov8-seg",
        variant="m",
        options=options,
    )
    converted = await sdk.resources.convert(
        "ultralytics/yolov8-pose",
        ArtifactFormat.COREML,
        variant="s",
        options=options,
    )

    assert pose.artifacts[0].available
    assert seg.artifacts[0].available
    assert converted.conversion_targets[0].available
    assert (tmp_path / "ultralytics" / "yolov8-pose" / "v8.2.0" / "s").exists()
    assert (tmp_path / "ultralytics" / "yolov8-seg" / "v8.2.0" / "m").exists()


@pytest.mark.parametrize(
    ("converter_type", "model_id", "filename"),
    [
        (YoloV8PoseConverter, "ultralytics/yolov8-pose", "yolov8n-pose.pt"),
        (YoloV8SegConverter, "ultralytics/yolov8-seg", "yolov8n-seg.pt"),
    ],
)
async def test_pose_and_seg_coreml_conversion_uses_raw_task_outputs(
    tmp_path: Path,
    monkeypatch: Any,
    converter_type: type[YoloV8PoseConverter] | type[YoloV8SegConverter],
    model_id: str,
    filename: str,
) -> None:
    export_arguments: dict[str, object] = {}

    class FakeYolo:
        def __init__(self, source: str) -> None:
            self.source = Path(source)

        def export(self, **kwargs: object) -> str:
            export_arguments.update(kwargs)
            output = self.source.with_suffix(".mlpackage")
            output.mkdir()
            (output / "model.mlmodel").write_bytes(b"coreml")
            return str(output)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    source_path = tmp_path / filename
    source_path.write_bytes(b"weights")
    source = ResolvedResource(
        path=source_path,
        source=UrlFileSource(url=f"https://models.example/{filename}"),
        digest="1" * 64,
        size_bytes=source_path.stat().st_size,
    )
    result = await converter_type().convert(
        ConversionRequest(
            model_id=model_id,
            model_revision="v8.2.0",
            variant="n",
            source=source,
            source_format=ArtifactFormat.PYTORCH,
            target_format=ArtifactFormat.COREML,
            output_path=tmp_path / "artifact.mlpackage",
            # The task converter must still suppress an invalid user override.
            options={"nms": True},
        )
    )

    assert result.path.is_dir()
    assert result.options["nms"] is False
    assert "variant" not in result.options
    assert export_arguments["format"] == "coreml"
    assert export_arguments["nms"] is False
