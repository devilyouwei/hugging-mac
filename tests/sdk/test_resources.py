from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.models.yolov8 import YOLOV8N_MANIFEST, register_yolov8
from hugging_mac_sdk.models.yolov8.assets import YoloV8AssetResolver
from hugging_mac_sdk.models.yolov8.config import YOLOV8N_SHA256, YoloV8InstanceConfig
from hugging_mac_sdk.models.yolov8.converter import YoloV8Converter
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.schemas.conversion import (
    ArtifactFormat,
    ConversionRequest,
    ConversionResult,
)
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, ResolvedResource


def _source() -> HuggingFaceSource:
    source = YOLOV8N_MANIFEST.resources[0]
    assert isinstance(source, HuggingFaceSource)
    return source


async def test_resolve_never_downloads_or_converts_missing_assets(tmp_path: Path) -> None:
    resolver = YoloV8AssetResolver(
        _source(),
        YoloV8InstanceConfig(model_home=tmp_path),
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
            "ultralytics/yolov8n",
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
            digest=YOLOV8N_SHA256,
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
    monkeypatch.setattr(
        "hugging_mac_sdk.models.yolov8.assets.file_sha256",
        lambda _: YOLOV8N_SHA256,
    )

    sdk = ModelSdk()
    from hugging_mac_sdk.converters import ConverterRegistry

    register_yolov8(sdk.registry, ConverterRegistry())
    options = {"model_home": tmp_path}

    initial = await sdk.resources.status("ultralytics/yolov8n", options=options)
    assert not initial.artifacts[0].available
    assert not initial.artifacts[1].available

    downloaded = await sdk.resources.download_source(
        "ultralytics/yolov8n",
        options=options,
    )
    assert downloaded.artifacts[0].available
    assert not downloaded.artifacts[1].available

    converted = await sdk.resources.convert(
        "ultralytics/yolov8n",
        ArtifactFormat.COREML,
        options=options,
    )
    assert converted.artifacts[0].available
    assert converted.artifacts[1].available
