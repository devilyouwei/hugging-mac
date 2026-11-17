from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hugging_mac_sdk import ModelSdk, ModelState, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.errors import InferenceError, ResourceIntegrityError, ResourceNotFoundError
from hugging_mac_sdk.models.d3 import D3_DEFINITION, D3_MANIFEST, register_d3
from hugging_mac_sdk.models.d3.config import D3InstanceConfig
from hugging_mac_sdk.models.d3.coreai import CoreAID3Engine
from hugging_mac_sdk.models.d3.definition import COREAI_LAYOUT, D3_CONFIG, DECISION_SUPPORT
from hugging_mac_sdk.models.d3.instance import D3Instance
from hugging_mac_sdk.models.d3.resources import D3ResourceResolver
from hugging_mac_sdk.schemas.decision import DecisionRequest

ARTIFACT = D3_CONFIG.get_artifact("coreai-fp32", variant="edge", runtime="coreai")
RESULT = {
    "model": "vllm-sr/d3-edge",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "shipping": 0.2},
        }
    },
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


def resolver(tmp_path: Path, **options: object) -> D3ResourceResolver:
    return D3ResourceResolver(
        D3InstanceConfig.model_validate({"model_home": tmp_path} | options),
        manifest=D3_MANIFEST,
        artifact=ARTIFACT,
        shared_artifacts=D3_DEFINITION.shared_artifacts,
    )


def populate(root: Path) -> None:
    for name in ARTIFACT.required_files:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    for share in D3_DEFINITION.shared_artifacts:
        shared_root = (
            root.parents[2] / "_shared" / share.artifact_id
            if root.name == ARTIFACT.artifact_id
            else root
        )
        for name in share.required_files:
            file = shared_root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("shared fixture")


def test_registration_does_not_import_frameworks() -> None:
    source = """
import sys
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.models.d3 import register_d3
sdk = ModelSdk()
definition = register_d3(sdk.registry)
instance = definition.runtime_factories['coreai']({})
assert instance.info().model_id == definition.manifest.model_id
assert not {'torch', 'numpy', 'transformers', 'coreai'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_manifest_runtime_and_capability_coverage() -> None:
    assert {runtime.name for runtime in D3_MANIFEST.runtimes} == set(
        D3_DEFINITION.runtime_factories
    )
    for variant in D3_MANIFEST.variants:
        for runtime, factory in D3_DEFINITION.runtime_factories.items():
            assert D3_CONFIG.get_artifacts(variant=variant.name, runtime=runtime)
            instance = factory({"variant": variant.name})
            assert instance.state is ModelState.CREATED
            assert set(instance.capabilities) == {DecisionMaking}
            assert D3_MANIFEST.capabilities == {"decision-making"}
    with pytest.raises(ResourceNotFoundError):
        D3_DEFINITION.runtime_factories["coreai"]({"variant": "unknown"})


async def test_resource_inventory_status_and_safe_delete(tmp_path: Path) -> None:
    resources = resolver(tmp_path)
    assert not resources.status().artifacts[0].available
    with pytest.raises(ResourceNotFoundError):
        await resources.resolve()
    populate(resources.path)
    assert await resources.resolve() == ARTIFACT.resolve(tmp_path)
    assert resources.status().artifacts[0].available
    assert "path" not in resources.status().model_dump_json()
    (resources.path / ARTIFACT.required_files[-1]).unlink()
    assert not resources.status().artifacts[0].available
    with pytest.raises(ResourceNotFoundError):
        await resources.resolve()
    await resources.delete()
    await resources.delete()
    external = tmp_path / "external"
    external.mkdir()
    with pytest.raises(ResourceIntegrityError):
        await resolver(tmp_path, artifact_path=external).delete()
    assert external.is_dir()


async def test_delete_rejects_symlink_escape(tmp_path: Path) -> None:
    resources = resolver(tmp_path / "managed")
    external = tmp_path / "external"
    populate(external)
    path = resources.path
    path.parent.mkdir(parents=True)
    path.symlink_to(external, target_is_directory=True)
    with pytest.raises(ResourceIntegrityError):
        await resources.delete()
    assert external.is_dir()


async def test_typed_capability_through_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CoreAID3Engine, "resolve", AsyncMock(return_value=Path("bundle")))
    load = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(CoreAID3Engine, "load", load)
    monkeypatch.setattr(CoreAID3Engine, "close", close)
    infer = AsyncMock(return_value=RESULT)
    monkeypatch.setattr(CoreAID3Engine, "infer", infer)
    sdk = ModelSdk()
    register_d3(sdk.registry)
    async with await sdk.acquire(
        D3_MANIFEST.model_id,
        runtime="coreai",
        reuse=ReusePolicy.DEDICATED,
    ) as handle:
        response = await handle.require(DecisionMaking).decide(
            DecisionRequest(
                state="duplicate charge",
                questions={
                    "route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}
                },
            )
        )
        assert response.model_dump(exclude_none=True)["answers"] == RESULT["answers"]
        assert response.runtime == "coreai"
        assert response.model_id == D3_MANIFEST.model_id
        assert response.device == "cpu"
        assert response.timings.inference_ms >= 0
        await handle.instance.load()
        load.assert_awaited_once()
        infer.side_effect = RuntimeError("native failure")
        with pytest.raises(InferenceError):
            await handle.require(DecisionMaking).decide(
                DecisionRequest(
                    state="",
                    questions={
                        "route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}
                    },
                )
            )
    close.assert_awaited_once()
    await handle.instance.unload()
    close.assert_awaited_once()


async def test_failed_load_closes_partial_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Avoid importing optional frameworks while exercising the real engine loader.
    monkeypatch.setitem(
        sys.modules,
        "hugging_mac_sdk.models.d3.utils.adapter",
        SimpleNamespace(
            create_runtime=lambda *args, **kwargs: None,
        ),
    )
    session = SimpleNamespace(close=AsyncMock())
    create = AsyncMock(side_effect=[session, RuntimeError("vision load failed")])
    monkeypatch.setattr("hugging_mac_sdk.models.d3.coreai.CoreAIProvider.create_session", create)
    resources = resolver(tmp_path)
    populate(resources.path)
    engine = CoreAID3Engine(
        D3InstanceConfig(), resources, COREAI_LAYOUT, model_name=RESULT["model"]
    )
    instance = D3Instance(D3InstanceConfig(), engine, D3_MANIFEST, DECISION_SUPPORT)
    with pytest.raises(Exception, match="Failed to load"):
        await instance.load()
    assert instance.state is ModelState.FAILED
    session.close.assert_awaited_once()
    assert engine._decision is engine._vision is engine._adapter is None


@pytest.mark.parametrize("worker_error", [False, True])
async def test_cancel_waits_for_worker_before_unload(tmp_path: Path, worker_error: bool) -> None:
    entered, release = Event(), Event()

    def system_one(**kwargs: object) -> dict:
        entered.set()
        assert release.wait(5)
        if worker_error:
            raise ValueError("worker failed after cancellation")
        return RESULT

    engine = CoreAID3Engine(
        D3InstanceConfig(), resolver(tmp_path), COREAI_LAYOUT, model_name=RESULT["model"]
    )
    engine._adapter = SimpleNamespace(system_one=system_one)
    instance = D3Instance(D3InstanceConfig(), engine, D3_MANIFEST, DECISION_SUPPORT)
    instance._state = ModelState.READY
    task = asyncio.create_task(
        instance.decide(
            DecisionRequest(
                state="",
                questions={
                    "route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}
                },
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        unload = asyncio.create_task(instance.unload())
        await asyncio.sleep(0)
        assert not task.done()
        assert not unload.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await unload
    assert engine._adapter is None


async def test_provider_reports_local_only_operations(tmp_path: Path) -> None:
    from hugging_mac_sdk.errors import UnsupportedRuntimeError
    from hugging_mac_sdk.schemas.conversion import ArtifactFormat

    provider = D3_DEFINITION.resource_provider
    assert provider is not None
    options = {"model_home": tmp_path, "runtime": "coreai"}
    status = await provider.status("edge", options)
    assert not status.artifacts[0].available
    for overwrite in (False, True):
        with pytest.raises(UnsupportedRuntimeError, match="local conversion"):
            await provider.download_source("edge", options, overwrite=overwrite)
        with pytest.raises(UnsupportedRuntimeError, match="cannot convert"):
            await provider.convert("edge", ArtifactFormat.ONNX, options, overwrite=overwrite)
    with pytest.raises(UnsupportedRuntimeError):
        await provider.delete("edge", options, runtime="mlx")
    populate(ARTIFACT.resolve(tmp_path))
    assert (await provider.status("edge", options)).artifacts[1].available
    assert not (await provider.delete("edge", options)).artifacts[0].available
    assert not (await provider.delete("edge", options)).artifacts[0].available


def test_declared_layout_assets_are_in_the_required_inventory() -> None:
    for asset in (COREAI_LAYOUT.decision_asset, COREAI_LAYOUT.vision_asset):
        assert asset / "main.mlirb" in ARTIFACT.required_files
    assert D3_MANIFEST.get_variant().metadata["response_model"] == RESULT["model"]


async def test_media_mapping_uses_typed_capability(tmp_path: Path) -> None:
    from io import BytesIO

    from hugging_mac_sdk.schemas.decision import DecisionVideoFrames
    from hugging_mac_sdk.schemas.detection import ImageInput
    from PIL import Image

    with BytesIO() as buffer:
        Image.new("RGB", (2, 2), (31, 67, 127)).save(buffer, format="PNG")
        encoded = buffer.getvalue()
    path = tmp_path / "frame.png"
    path.write_bytes(encoded)
    infer = AsyncMock(return_value=RESULT)
    engine = SimpleNamespace(runtime_name="coreai", device="cpu", infer=infer)
    instance = D3Instance(D3InstanceConfig(), engine, D3_MANIFEST, DECISION_SUPPORT)
    request = DecisionRequest(
        state="media",
        questions={"route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}},
        images=(ImageInput(data=encoded),),
        videos=(tmp_path / "clip.mp4", DecisionVideoFrames(frames=(ImageInput(path=path),))),
    )
    with pytest.raises(InferenceError, match="Load"):
        await instance.require(DecisionMaking).decide(request)
    infer.assert_not_called()
    instance._state = ModelState.READY
    response = await instance.require(DecisionMaking).decide(request)
    inputs = infer.call_args.kwargs
    assert inputs["images"][0].getpixel((0, 0)) == (31, 67, 127)
    assert inputs["videos"][0] == str(tmp_path / "clip.mp4")
    assert inputs["videos"][1][0].getpixel((0, 0)) == (31, 67, 127)
    assert response.timings.preprocess_ms >= 0
    assert response.timings.postprocess_ms >= 0


async def test_pytorch_capability_and_failed_load_cleanup(tmp_path, monkeypatch) -> None:

    model = SimpleNamespace(
        device="cpu",
        image_unavailable=None,
        video_unavailable=None,
        system_one=lambda **kwargs: RESULT,
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.d3.torch.load_upstream",
        lambda root: SimpleNamespace(D3=lambda *args, **kwargs: model),
    )
    for artifact in D3_CONFIG.get_artifacts(variant="edge", runtime="pytorch", include_shared=True):
        for name in artifact.required_files:
            file = tmp_path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture")
    sdk = ModelSdk()
    register_d3(sdk.registry)
    async with await sdk.acquire(
        D3_MANIFEST.model_id,
        runtime="pytorch",
        reuse=ReusePolicy.DEDICATED,
        options={"source_path": tmp_path},
    ) as handle:
        result = await handle.require(DecisionMaking).decide(
            DecisionRequest(
                state="",
                questions={
                    "route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}
                },
            )
        )
        assert result.runtime == "pytorch"
        assert result.model_dump(exclude_none=True)["answers"] == RESULT["answers"]
        engine = handle.instance._engine
    assert engine._model is None
    model.video_unavailable = "missing processor"
    failed = D3_DEFINITION.runtime_factories["pytorch"]({"source_path": tmp_path})
    with pytest.raises(Exception, match="Failed to load"):
        await failed.load()
    assert failed._engine._model is None


async def test_source_download_and_conversion_resource_dispatch(tmp_path, monkeypatch) -> None:
    from hugging_mac_sdk.models.d3.definition import D3_CONVERTER
    from hugging_mac_sdk.schemas.conversion import ArtifactFormat

    source = D3_CONFIG.get_artifact(shared=False, variant="edge", runtime="pytorch")
    sdk = ModelSdk()
    register_d3(sdk.registry)
    provider = D3_DEFINITION.resource_provider
    calls = []

    async def download(self, declaration, destination, **kwargs):
        calls.append(kwargs)
        for name in declaration.allow_patterns:
            file = destination / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("source")

    monkeypatch.setattr("hugging_mac_sdk.models.d3.resources.ResourceDownloader.download", download)
    options = {"model_home": tmp_path}
    status = await sdk.resources.download_source(D3_MANIFEST.model_id, options=options)
    assert status.artifacts[0].available
    assert status.conversion_targets[0].runtime == "coreai"
    await sdk.resources.download_source(D3_MANIFEST.model_id, options=options)
    assert len(calls) == 4
    await sdk.resources.download_source(D3_MANIFEST.model_id, options=options, overwrite=True)
    assert calls[-1]["overwrite"] is True

    async def validate_view(request):
        assert (request.source.path / "tokenizer.json").is_symlink()
        assert (request.source.path / "model-00001-of-00001.safetensors").resolve() == (
            source.resolve(tmp_path) / "model-00001-of-00001.safetensors"
        )

    convert = AsyncMock(side_effect=validate_view)
    monkeypatch.setattr(D3_CONVERTER, "convert", convert)
    await provider.convert("edge", ArtifactFormat.COREAI, options, overwrite=True)
    request = convert.call_args.args[0]
    assert not request.source.path.exists()  # temporary conversion view is closed
    assert request.output_path == ARTIFACT.resolve(tmp_path)
    assert request.overwrite


async def test_shared_inventory_status_and_runtime_deletion(tmp_path) -> None:
    source = D3_CONFIG.get_artifact(shared=False, variant="edge", runtime="pytorch")
    for name in source.required_files:
        file = source.resolve(tmp_path) / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("weights")
    populate(ARTIFACT.resolve(tmp_path))
    sdk = ModelSdk()
    register_d3(sdk.registry)
    options = {"model_home": tmp_path}
    status = await sdk.resources.status(D3_MANIFEST.model_id, options=options)
    assert all(runtime.available for runtime in status.runtimes)
    shares = D3_DEFINITION.required_shared_artifacts(variant="edge", runtime="pytorch")
    assert {a.artifact_id for a in shares} == {"tokenizer", "processor", "runtime"}
    for share in shares:
        for scoped in (source, ARTIFACT):
            assert not set(share.required_files) & set(scoped.required_files)
            assert not any(
                (scoped.resolve(tmp_path) / name).exists() for name in share.required_files
            )
    tokenizer = D3_CONFIG.get_artifact("tokenizer", shared=True)
    file = tokenizer.resolve(tmp_path) / tokenizer.required_files[0]
    file.unlink()
    status = await sdk.resources.status(D3_MANIFEST.model_id, options=options)
    assert not any(runtime.available for runtime in status.runtimes)
    with pytest.raises(ResourceNotFoundError, match="shared resource"):
        await resolver(tmp_path).resolve()
    file.write_text("restored")
    with resolver(tmp_path).view() as view:
        assert (view / "tokenizer.json").is_symlink()
        assert (view / "tokenizer.json").resolve() == file
    assert not view.exists()
    await sdk.resources.delete(D3_MANIFEST.model_id, runtime="coreai", options=options)
    status = await sdk.resources.status(D3_MANIFEST.model_id, options=options)
    assert next(r for r in status.runtimes if r.runtime == "pytorch").available
    assert all(share.resolve(tmp_path).is_dir() for share in shares)


async def test_coreai_keeps_shared_view_alive_until_unload(tmp_path, monkeypatch) -> None:
    resources = resolver(tmp_path)
    populate(resources.path)
    views = []

    def create_runtime(root, *args, **kwargs):
        views.append(root)

        def infer(**inputs):
            assert (root / "tokenizer.json").read_text() == "shared fixture"
            return RESULT

        return SimpleNamespace(system_one=infer)

    monkeypatch.setitem(
        sys.modules,
        "hugging_mac_sdk.models.d3.utils.adapter",
        SimpleNamespace(create_runtime=create_runtime),
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.d3.coreai.CoreAIProvider.create_session",
        AsyncMock(
            side_effect=[
                SimpleNamespace(device="cpu", close=AsyncMock()),
                SimpleNamespace(device="cpu", close=AsyncMock()),
            ]
        ),
    )
    engine = CoreAID3Engine(
        D3InstanceConfig(runtime="coreai"), resources, COREAI_LAYOUT, model_name=RESULT["model"]
    )
    instance = D3Instance(D3InstanceConfig(runtime="coreai"), engine, D3_MANIFEST, DECISION_SUPPORT)
    await instance.load()
    assert views[0].is_dir()
    await instance.require(DecisionMaking).decide(
        DecisionRequest(
            state="",
            questions={
                "route": {"type": "choice", "criteria": {"billing": None, "shipping": None}}
            },
        )
    )
    await instance.unload()
    assert not views[0].exists()
