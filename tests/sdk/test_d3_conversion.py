from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

import pytest
from hugging_mac_sdk import ConverterRegistry, ModelRegistry
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.models.d3 import register_d3
from hugging_mac_sdk.models.d3.converter import D3CoreAIConverter
from hugging_mac_sdk.models.d3.definition import COREAI_LAYOUT, D3_CONFIG, D3_MANIFEST
from hugging_mac_sdk.schemas.conversion import ArtifactFormat, ConversionRequest
from hugging_mac_sdk.schemas.resources import ResolvedResource

SOURCE = D3_CONFIG.get_artifact(shared=False, variant="edge", runtime="pytorch")
TARGET = D3_CONFIG.get_artifact(shared=False, variant="edge", runtime="coreai")


def request(tmp_path: Path, *, overwrite: bool = False) -> ConversionRequest:
    source = SOURCE.resolve(tmp_path)
    names = tuple(
        name
        for artifact in D3_CONFIG.get_artifacts(
            variant="edge", runtime="pytorch", include_shared=True
        )
        for name in artifact.required_files
    )
    for name in names:
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("source fixture")
    return ConversionRequest(
        source=ResolvedResource(path=source, source=SOURCE.source, size_bytes=0),
        source_format=ArtifactFormat.SAFETENSORS,
        target_format=ArtifactFormat.COREAI,
        output_path=TARGET.resolve(tmp_path),
        model_id=D3_MANIFEST.model_id,
        variant="edge",
        overwrite=overwrite,
    )


def build(source: Path, output: Path) -> None:
    for asset in (COREAI_LAYOUT.decision_asset, COREAI_LAYOUT.vision_asset):
        for name in ("main.mlirb", "main.hash", "metadata.json"):
            file = output / asset / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("native fixture")


async def test_converter_registration_and_complete_atomic_bundle(tmp_path, monkeypatch):
    converters = ConverterRegistry()
    definition = register_d3(ModelRegistry(), converters)
    conversion = request(tmp_path)
    converter = converters.resolve(conversion, preferred_ids=definition.converter_ids)
    monkeypatch.setattr(converter, "_build", build)
    result = await converter.convert(conversion)
    assert all((result.path / name).is_file() for name in TARGET.required_files)
    for share in D3_CONFIG.artifacts:
        if share.shared:
            assert not any((result.path / name).exists() for name in share.required_files)
    assert result.digest and result.size_bytes > 0
    assert result.path == TARGET.resolve(tmp_path)
    with pytest.raises(FileExistsError):
        await converter.convert(conversion)
    await converter.convert(conversion.model_copy(update={"overwrite": True}))
    with pytest.raises(ValueError, match="overlap"):
        await converter.convert(
            conversion.model_copy(update={"output_path": conversion.source.path, "overwrite": True})
        )


async def test_failed_conversion_keeps_existing_bundle(tmp_path, monkeypatch):
    converter = D3CoreAIConverter(D3_CONFIG, COREAI_LAYOUT)
    conversion = request(tmp_path, overwrite=True)
    conversion.output_path.mkdir(parents=True)
    marker = conversion.output_path / "keep"
    marker.write_text("old bundle")
    monkeypatch.setattr(converter, "_build", lambda source, output: None)
    with pytest.raises(ResourceNotFoundError, match="incomplete native"):
        await converter.convert(conversion)
    assert marker.read_text() == "old bundle"
    assert not list(conversion.output_path.parent.glob(".d3-coreai-*"))


async def test_cancelled_conversion_does_not_publish(tmp_path, monkeypatch):
    converter = D3CoreAIConverter(D3_CONFIG, COREAI_LAYOUT)
    conversion = request(tmp_path)
    entered, release = Event(), Event()

    def wait_build(source, output):
        entered.set()
        assert release.wait(5)
        build(source, output)

    monkeypatch.setattr(converter, "_build", wait_build)
    task = asyncio.create_task(converter.convert(conversion))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not conversion.output_path.exists()
    assert not list(conversion.output_path.parent.glob(".d3-coreai-*"))
