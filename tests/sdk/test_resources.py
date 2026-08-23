from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.errors import ResourceIntegrityError, ResourceNotFoundError
from hugging_mac_sdk.models.yolov8 import YOLOV8_MANIFEST, register_yolov8
from hugging_mac_sdk.models.yolov8.config import YoloV8InstanceConfig
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.models.yolov8.definition import YOLOV8_CONFIG, YOLOV8_CONVERTER
from hugging_mac_sdk.models.yolov8.resources import YoloV8ResourceResolver
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, ResolvedResource


def _source() -> HuggingFaceSource:
    source = YOLOV8_CONFIG.get_artifact("source", variant="n", runtime="pytorch-mps").source
    assert isinstance(source, HuggingFaceSource)
    return source


async def test_resolve_never_downloads_or_converts_missing_assets(tmp_path: Path) -> None:
    resolver = YoloV8ResourceResolver(
        _source(),
        YoloV8InstanceConfig(model_home=tmp_path),
        YOLOV8_MANIFEST,
        {artifact.artifact_id: artifact for artifact in YOLOV8_CONFIG.get_artifacts(variant="n")},
        converter=YOLOV8_CONVERTER,
    )

    with pytest.raises(ResourceNotFoundError, match="not downloaded"):
        await resolver.resolve_source()
    with pytest.raises(ResourceNotFoundError, match="has not been converted"):
        await resolver.resolve_coreml()

    assert tuple(tmp_path.rglob("*")) == ()

    sdk = ModelSdk()
    from hugging_mac_sdk.converters import ConverterRegistry

    register_yolov8(sdk.registry, ConverterRegistry())
    with pytest.raises(ResourceNotFoundError, match="not downloaded"):
        await sdk.acquire(
            "ultralytics/yolov8",
            runtime="pytorch-mps",
            options={"model_home": tmp_path},
        )
    assert tuple(tmp_path.rglob("*")) == ()


async def test_sdk_exposes_explicit_download_and_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_download(
        _: ResourceDownloader,
        source: Any,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"explicitly downloaded")
        return ResolvedResource(
            path=destination,
            source=source,
            digest="downloaded",
            size_bytes=destination.stat().st_size,
        )

    async def fake_convert(
        _: YoloV8Converter,
        request: ConversionRequest,
    ) -> ConversionResult:
        request.output_path.mkdir(parents=True)
        (request.output_path / "model.mlmodel").write_bytes(b"coreml")
        return ConversionResult(
            path=request.output_path,
            format=request.target_format,
            digest="0" * 64,
            size_bytes=6,
            converter_id="ultralytics.yolov8",
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    monkeypatch.setattr(YoloV8Converter, "convert", fake_convert)

    sdk = ModelSdk()
    from hugging_mac_sdk.converters import ConverterRegistry

    register_yolov8(sdk.registry, ConverterRegistry())
    options = {"model_home": tmp_path}

    initial = await sdk.resources.status("ultralytics/yolov8", options=options)
    assert not initial.artifacts[0].available
    assert not initial.artifacts[1].available
    assert len(initial.conversion_targets) == 2
    assert initial.conversion_targets[0].target_format == "coreml"
    assert not initial.conversion_targets[0].available

    downloaded = await sdk.resources.download_source(
        "ultralytics/yolov8",
        options=options,
    )
    assert downloaded.artifacts[0].available
    assert not downloaded.artifacts[1].available

    converted = await sdk.resources.convert(
        "ultralytics/yolov8",
        ArtifactFormat.COREML,
        options=options,
    )
    assert converted.artifacts[0].available
    assert converted.artifacts[1].available
    assert converted.conversion_targets[0].available
    assert converted.conversion_targets[0].runtime == "coreml"
    assert converted.conversion_targets[0].size_bytes == 6
    assert converted.total_size_bytes == sum(
        artifact.size_bytes or 0 for artifact in converted.artifacts
    )
    runtime_sizes = {runtime.runtime: runtime for runtime in converted.runtimes}
    assert runtime_sizes["pytorch-mps"].available
    assert runtime_sizes["pytorch-mps"].size_bytes > 0
    assert runtime_sizes["coreml"].available
    assert runtime_sizes["coreml"].size_bytes == 6

    without_coreml = await sdk.resources.delete(
        "ultralytics/yolov8",
        runtime="coreml",
        options=options,
    )
    assert without_coreml.artifacts[0].available
    assert not without_coreml.artifacts[1].available

    empty = await sdk.resources.delete(
        "ultralytics/yolov8",
        options=options,
    )
    assert not any(artifact.available for artifact in empty.artifacts)
    assert empty.total_size_bytes == 0


async def test_delete_refuses_custom_artifact_outside_model_root(tmp_path: Path) -> None:
    outside = tmp_path / "external.pt"
    outside.write_bytes(b"keep me")
    resolver = YoloV8ResourceResolver(
        _source(),
        YoloV8InstanceConfig(
            model_home=tmp_path / "managed",
            source_path=outside,
            runtime="pytorch-mps",
        ),
        YOLOV8_MANIFEST,
        {artifact.artifact_id: artifact for artifact in YOLOV8_CONFIG.get_artifacts(variant="n")},
        converter=YOLOV8_CONVERTER,
    )

    with pytest.raises(ResourceIntegrityError, match="outside"):
        await resolver.delete(runtime="pytorch-mps")

    assert outside.read_bytes() == b"keep me"


async def test_yolov8_variant_resources_are_downloaded_and_deleted_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_download(
        _: ResourceDownloader,
        source: HuggingFaceSource,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.filename.encode() if source.filename else b"weights")
        return ResolvedResource(
            path=destination,
            source=source,
            digest="downloaded",
            size_bytes=destination.stat().st_size,
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    sdk = ModelSdk()
    from hugging_mac_sdk.converters import ConverterRegistry

    register_yolov8(sdk.registry, ConverterRegistry())
    options = {"model_home": tmp_path}

    for variant in ("n", "s", "m"):
        status = await sdk.resources.download_source(
            "ultralytics/yolov8",
            variant=variant,
            options=options,
        )
        assert status.variant == variant
        assert status.artifacts[0].available
        assert (
            tmp_path
            / "ultralytics"
            / "yolov8"
            / variant
            / "pytorch-mps"
            / "source"
            / f"yolov8{variant}.pt"
        ).is_file()

    await sdk.resources.delete(
        "ultralytics/yolov8",
        variant="s",
        options=options,
    )
    assert not (tmp_path / "ultralytics" / "yolov8" / "s").exists()
    assert (tmp_path / "ultralytics" / "yolov8" / "n").exists()
    assert (tmp_path / "ultralytics" / "yolov8" / "m").exists()
