from __future__ import annotations

from pathlib import Path

import pytest
from hugging_mac_sdk.converters.base import ModelConverter
from hugging_mac_sdk.converters.registry import ConversionService, ConverterRegistry
from hugging_mac_sdk.converters.yolov8 import YoloV8ExportOptions
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.yolov8 import YOLOV8_MANIFEST, register_yolov8
from hugging_mac_sdk.models.yolov8.config import YOLOV8_SHA256
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)
from hugging_mac_sdk.schemas.resources import (
    HuggingFaceSource,
    ResolvedResource,
    UrlFileSource,
)

YOLOV8_N_SHA256 = YOLOV8_SHA256["n"]


class FakeConverter(ModelConverter):
    def __init__(self, converter_id: str, *, priority: int = 0) -> None:
        self._converter_id = converter_id
        self._priority = priority

    @property
    def converter_id(self) -> str:
        return self._converter_id

    @property
    def priority(self) -> int:
        return self._priority

    def supports(self, request: ConversionRequest) -> bool:
        return request.source_format is ArtifactFormat.PYTORCH

    async def convert(self, request: ConversionRequest) -> ConversionResult:
        return ConversionResult(
            path=request.output_path,
            format=request.target_format,
            digest="0" * 64,
            size_bytes=0,
            converter_id=self.converter_id,
        )


def resource(path: Path) -> ResolvedResource:
    return ResolvedResource(
        path=path,
        source=UrlFileSource(url="https://models.example/yolov8n.pt"),
        digest="1" * 64,
        size_bytes=path.stat().st_size,
    )


async def test_converter_registry_prefers_model_binding(tmp_path: Path) -> None:
    source_path = tmp_path / "model.pt"
    source_path.write_bytes(b"weights")
    request = ConversionRequest(
        source=resource(source_path),
        source_format=ArtifactFormat.PYTORCH,
        target_format=ArtifactFormat.ONNX,
        output_path=tmp_path / "model.onnx",
    )
    registry = ConverterRegistry()
    registry.register(FakeConverter("generic", priority=100))
    registry.register(FakeConverter("model-specific"))

    selected = registry.resolve(request, preferred_ids=("model-specific",))

    assert selected.converter_id == "model-specific"


async def test_yolov8_registration_and_coreml_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_options: YoloV8ExportOptions | None = None

    def fake_export(
        _: YoloV8Converter,
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

    monkeypatch.setattr(YoloV8Converter, "_export", fake_export)
    source_path = tmp_path / "yolov8n.pt"
    source_path.write_bytes(b"trusted test weights")
    source = ResolvedResource(
        path=source_path,
        source=HuggingFaceSource(
            repo_id="Ultralytics/YOLOv8",
            revision="8a9e1a5",
            filename="yolov8n.pt",
            expected_sha256=YOLOV8_N_SHA256,
        ),
        digest="2" * 64,
        size_bytes=source_path.stat().st_size,
    )

    models = ModelRegistry()
    converters = ConverterRegistry()
    definition = register_yolov8(models, converters)
    request = ConversionRequest(
        model_id="ultralytics/yolov8",
        model_revision=definition.manifest.revision,
        variant="n",
        source=source,
        source_format=ArtifactFormat.PYTORCH,
        target_format=ArtifactFormat.COREML,
        output_path=tmp_path / "artifacts" / "yolov8n.mlpackage",
    )

    result = await ConversionService(converters).convert(request, definition=definition)

    assert result.converter_id == "ultralytics.yolov8"
    assert result.model_id == "ultralytics/yolov8"
    assert result.variant == "n"
    assert (result.path / "model.mlmodel").read_bytes() == b"coreml"
    assert export_options is not None
    assert export_options.imgsz == 640
    assert export_options.batch == 1
    assert export_options.half is True
    assert export_options.quantize == 16
    assert export_options.nms is False
    assert export_options.dynamic is False


def test_yolov8_manifest_pins_source_and_has_runtime_factory() -> None:
    source = YOLOV8_MANIFEST.get_variant("n").resources[0]

    assert isinstance(source, HuggingFaceSource)
    assert source.repo_id == "Ultralytics/YOLOv8"
    assert source.revision == "8a9e1a5"
    assert source.filename == "yolov8n.pt"
    assert source.expected_sha256 == YOLOV8_N_SHA256
    assert YOLOV8_MANIFEST.license == "AGPL-3.0"
    assert tuple(item.name for item in YOLOV8_MANIFEST.variants) == ("n", "s", "m")
    assert YOLOV8_MANIFEST.default_variant == "n"

    models = ModelRegistry()
    converters = ConverterRegistry()
    definition = register_yolov8(models, converters)
    assert definition.factory is None
    assert set(definition.runtime_factories) == {"pytorch-mps", "coreml", "onnx"}
