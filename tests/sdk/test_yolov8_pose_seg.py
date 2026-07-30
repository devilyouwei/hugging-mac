from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import InstanceSegmentation, PoseEstimation
from hugging_mac_sdk.converters import ConverterRegistry
from hugging_mac_sdk.converters.yolov8 import YoloV8ExportOptions
from hugging_mac_sdk.models.yolov8_pose import (
    YOLOV8_POSE_MANIFEST,
    register_yolov8_pose,
)
from hugging_mac_sdk.models.yolov8_pose.config import YoloV8PoseInstanceConfig
from hugging_mac_sdk.models.yolov8_pose.converter import YoloV8PoseConverter
from hugging_mac_sdk.models.yolov8_pose.instance import YoloV8PoseInstance
from hugging_mac_sdk.models.yolov8_pose.torch import TorchYoloV8PoseEngine
from hugging_mac_sdk.models.yolov8_seg import (
    YOLOV8_SEG_MANIFEST,
    register_yolov8_seg,
)
from hugging_mac_sdk.models.yolov8_seg.config import YoloV8SegInstanceConfig
from hugging_mac_sdk.models.yolov8_seg.converter import YoloV8SegConverter
from hugging_mac_sdk.models.yolov8_seg.coreml import CoreMlYoloV8SegEngine
from hugging_mac_sdk.models.yolov8_seg.instance import YoloV8SegInstance
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest, ConversionResult
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput
from hugging_mac_sdk.schemas.resources import ResolvedResource, UrlFileSource
from hugging_mac_sdk.schemas.segmentation import PolygonPoint
from PIL import Image


class FakeSession:
    def __init__(self, outputs: dict[str, Any], device: str) -> None:
        self.outputs = outputs
        self.device = device

    def run(self, _: dict[str, Any]) -> dict[str, Any]:
        return self.outputs

    async def close(self) -> None:
        return None


class FakeAssets:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def resolve_source(self) -> ResolvedResource:
        return self._resource()

    async def resolve_coreml(self) -> ResolvedResource:
        return self._resource()

    async def resolve_onnx(self) -> ResolvedResource:
        return self._resource()

    def _resource(self) -> ResolvedResource:
        return ResolvedResource(
            path=self._path,
            source=UrlFileSource(url="https://models.example/yolo.pt"),
            size_bytes=0,
        )


def pose_outputs() -> dict[str, Any]:
    prediction = np.zeros((1, 56, 8400), dtype=np.float32)
    prediction[0, :4, 0] = (60, 200, 100, 200)
    prediction[0, 4, 0] = 0.9
    prediction[0, 5:11, 0] = (15, 105, 0.8, 20, 110, 0.7)
    return {"predictions": prediction}


def seg_outputs() -> dict[str, Any]:
    prediction = np.zeros((1, 116, 8400), dtype=np.float32)
    prediction[0, :4, 0] = (60, 200, 100, 200)
    prediction[0, 4, 0] = 0.9
    prediction[0, 84, 0] = 10
    prototypes = np.zeros((1, 32, 160, 160), dtype=np.float32)
    prototypes[0, 0] = 1
    return {"predictions": prediction, "prototypes": prototypes}


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
        pose_instance = sdk.registry.create_instance(
            "ultralytics/yolov8-pose",
            variant=variant,
            runtime="coreml",
        )
        seg_instance = sdk.registry.create_instance(
            "ultralytics/yolov8-seg",
            variant=variant,
            runtime="pytorch-mps",
        )
        assert isinstance(pose_instance, YoloV8PoseInstance)
        assert isinstance(seg_instance, YoloV8SegInstance)
        assert pose_instance.info().runtime == "coreml"
        assert seg_instance.info().runtime == "pytorch-mps"


async def test_pose_and_seg_instances_map_task_specific_results(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    image = tmp_path / "input.jpg"
    Image.new("RGB", (640, 480)).save(image)
    pose_weights = tmp_path / "pose.pt"
    pose_weights.write_bytes(b"weights")
    seg_package = tmp_path / "seg.mlpackage"
    seg_package.mkdir()

    async def fake_torch_session(*_: Any, **__: Any) -> FakeSession:
        return FakeSession(pose_outputs(), "mps")

    async def fake_coreml_session(*_: Any, **__: Any) -> FakeSession:
        return FakeSession(seg_outputs(), "all")

    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8_pose.torch.TorchProvider.create_session",
        fake_torch_session,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8_pose.torch.torch_input",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8_seg.coreml.CoreMLProvider.create_session",
        fake_coreml_session,
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8_seg.utils.postprocess.masks_to_polygons",
        lambda *_: (
            (
                (
                    PolygonPoint(x=10, y=20),
                    PolygonPoint(x=110, y=20),
                    PolygonPoint(x=110, y=220),
                ),
            ),
        ),
    )

    pose_config = YoloV8PoseInstanceConfig(runtime="pytorch-mps", variant="s")
    pose_engine = TorchYoloV8PoseEngine(
        pose_config,
        FakeAssets(pose_weights),  # type: ignore[arg-type]
    )
    pose = YoloV8PoseInstance(pose_config, pose_engine)
    seg_config = YoloV8SegInstanceConfig(runtime="coreml", variant="m")
    seg_engine = CoreMlYoloV8SegEngine(
        seg_config,
        FakeAssets(seg_package),  # type: ignore[arg-type]
    )
    seg = YoloV8SegInstance(seg_config, seg_engine)
    request = DetectionRequest(image=ImageInput(path=image))

    await pose.load()
    pose_response = await pose.require(PoseEstimation).estimate_pose(request)  # type: ignore[type-abstract]
    await seg.load()
    seg_response = await seg.require(InstanceSegmentation).segment(request)  # type: ignore[type-abstract]

    assert pose_response.poses[0].keypoints[1].confidence == pytest.approx(0.7)
    assert pose_response.poses[0].label == "person"
    assert seg_response.segments[0].polygons
    assert all(
        10 <= point.x <= 110 and 20 <= point.y <= 220
        for point in seg_response.segments[0].polygons[0]
    )
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
    export_options: YoloV8ExportOptions | None = None

    def fake_export(
        _: YoloV8PoseConverter | YoloV8SegConverter,
        source: Path,
        target_format: ArtifactFormat,
        options: YoloV8ExportOptions,
    ) -> Path:
        nonlocal export_options
        assert target_format is ArtifactFormat.COREML
        export_options = options
        output = source.with_suffix(".mlpackage")
        output.mkdir()
        (output / "model.mlmodel").write_bytes(b"coreml")
        return output

    monkeypatch.setattr(converter_type, "_export", fake_export)
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
    assert export_options is not None
    assert export_options.nms is False
