from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hugging_mac_sdk import ModelSdk, ModelState, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.errors import InferenceError, ResourceIntegrityError, ResourceNotFoundError
from hugging_mac_sdk.models.vela import VELA_DEFINITION, VELA_MANIFEST, register_vela
from hugging_mac_sdk.models.vela.config import VelaInstanceConfig
from hugging_mac_sdk.models.vela.coreai import CoreAIVelaEngine
from hugging_mac_sdk.models.vela.definition import COREAI_LAYOUT, VELA_CONFIG
from hugging_mac_sdk.models.vela.resources import VelaResourceResolver
from hugging_mac_sdk.schemas.decision import DecisionRequest

ARTIFACT = VELA_CONFIG.get_artifact("coreai-fp32", variant="3-fast", runtime="coreai")
RESULT = {
    "model": "vllm-sr/vela3-fast",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "shipping": 0.2},
        }
    },
    "usage": {"input_tokens": 12, "output_tokens": 0},
}


def resolver(tmp_path: Path, **options: object) -> VelaResourceResolver:
    return VelaResourceResolver(
        VelaInstanceConfig.model_validate({"model_home": tmp_path} | options),
        manifest=VELA_MANIFEST,
        artifact=ARTIFACT,
        shared_artifacts=VELA_DEFINITION.shared_artifacts,
    )


def populate(root: Path) -> None:
    for name in ARTIFACT.required_files:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    for share in VELA_DEFINITION.shared_artifacts:
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
from hugging_mac_sdk.models.vela import register_vela
sdk = ModelSdk()
definition = register_vela(sdk.registry)
instance = definition.runtime_factories['coreai']({})
assert instance.info().model_id == definition.manifest.model_id
assert not {'torch', 'numpy', 'transformers', 'coreai'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_manifest_runtime_and_capability_coverage() -> None:
    assert {runtime.name for runtime in VELA_MANIFEST.runtimes} == set(
        VELA_DEFINITION.runtime_factories
    )
    for variant in VELA_MANIFEST.variants:
        for runtime, factory in VELA_DEFINITION.runtime_factories.items():
            assert VELA_CONFIG.get_artifacts(variant=variant.name, runtime=runtime)
            instance = factory({"variant": variant.name})
            assert instance.state is ModelState.CREATED
            assert set(instance.capabilities) == {DecisionMaking}
            assert VELA_MANIFEST.capabilities == {"decision-making"}
    with pytest.raises(ResourceNotFoundError):
        VELA_DEFINITION.runtime_factories["coreai"]({"variant": "unknown"})


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
    monkeypatch.setattr(CoreAIVelaEngine, "resolve", AsyncMock(return_value=Path("bundle")))
    load = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(CoreAIVelaEngine, "load", load)
    monkeypatch.setattr(CoreAIVelaEngine, "close", close)
    infer = AsyncMock(return_value=RESULT)
    monkeypatch.setattr(CoreAIVelaEngine, "infer", infer)
    sdk = ModelSdk()
    register_vela(sdk.registry)
    async with await sdk.acquire(
        VELA_MANIFEST.model_id,
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
        assert response.model_id == VELA_MANIFEST.model_id
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


@pytest.mark.parametrize("runtime", ["pytorch", "coreai"])
async def test_sdk_download_shares_overwrite_and_delete(tmp_path, monkeypatch, runtime):
    calls = []

    async def download(self, source, output, **kwargs):
        artifact = next(a for a in VELA_CONFIG.artifacts if a.source == source)
        for name in artifact.required_files:
            file = output / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture")
        calls.append((artifact.artifact_id, kwargs["overwrite"]))

    monkeypatch.setattr(
        "hugging_mac_sdk.resources.downloader.ResourceDownloader.download", download
    )
    sdk = ModelSdk()
    register_vela(sdk.registry)
    options = {"model_home": tmp_path, "runtime": runtime}
    status = await sdk.resources.download_source(VELA_MANIFEST.model_id, options=options)
    selected = VELA_CONFIG.get_artifact(shared=False, variant="3-fast", runtime=runtime)
    assert (selected.artifact_id, False) in calls
    assert ("tokenizer", False) in calls
    assert all(a.available for a in status.artifacts if a.shared or a.runtime == runtime)
    await sdk.resources.download_source(VELA_MANIFEST.model_id, options=options)
    assert len(calls) == 2
    await sdk.resources.download_source(VELA_MANIFEST.model_id, options=options, overwrite=True)
    assert (selected.artifact_id, True) in calls
    await sdk.resources.delete(VELA_MANIFEST.model_id, runtime=runtime, options=options)
    assert not selected.resolve(tmp_path).exists()


def test_normalize_shared_choice_criteria():
    from hugging_mac_sdk.models.vela.utils.preprocess import prepare_request

    inputs = prepare_request(
        DecisionRequest(
            state={"a": 1},
            questions={"q": {"type": "choice", "criteria": ["yes", "no"]}},
        )
    )
    assert inputs["questions"]["q"]["criteria"] == {"yes": None, "no": None}


async def test_failed_native_load_releases_view(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "hugging_mac_sdk.models.vela.utils.packing.DecisionPacker",
        lambda root: SimpleNamespace(manifest={"function": "main"}),
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.vela.coreai.CoreAIProvider.create_session",
        AsyncMock(side_effect=RuntimeError("native load failed")),
    )
    resources = resolver(tmp_path)
    populate(resources.path)
    engine = CoreAIVelaEngine(
        VelaInstanceConfig(), resources, COREAI_LAYOUT, model_name="vllm-sr/vela3-fast"
    )
    with pytest.raises(RuntimeError, match="native load"):
        await engine.load(resources.path)
    assert engine._session is None and engine._packer is None
    assert not engine._views._exit_callbacks
    await engine.close()


async def test_cancelled_worker_is_drained():
    import asyncio
    from threading import Event

    from hugging_mac_sdk.models.vela.utils.execution import run_blocking

    started, release = Event(), Event()

    def work():
        started.set()
        assert release.wait(5)

    task = asyncio.create_task(run_blocking(work))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_packing_json_markers_and_exact_capacity():
    import json
    from types import SimpleNamespace

    import numpy as np
    from hugging_mac_sdk.models.vela.utils.packing import DecisionPacker

    encoded = []

    def encode(text, **kwargs):
        encoded.append(text)
        return SimpleNamespace(ids=[7] * len(text))

    packer = DecisionPacker.__new__(DecisionPacker)
    packer.tokenizer = SimpleNamespace(encode=encode)
    packer.config = {
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "marker_ids": {"[Q]": 1, "[O]": 2, "[SEP_SCHEMA]": 3, "[SEG_user]": 4},
    }
    packer.manifest = {"sequence_length": 1000, "dynamic_sequence_length": True}
    question = {
        "type": "choice",
        "instructions": {"task": "route"},
        "criteria": {"a": None, "b": {"kind": "billing"}},
    }
    arrays, labels = packer.pack({"message": "中文"}, question)
    assert labels == ["a", "b"]
    assert json.dumps({"message": "中文"}, ensure_ascii=False, separators=(",", ":")) in encoded
    assert 'b: {"kind":"billing"}' in encoded
    assert arrays["input_ids"].dtype == np.int32
    assert arrays["pool_mask"].dtype == np.float32
    assert arrays["input_ids"][0, arrays["option_positions"][0]].tolist() == [2, 2]
    width = arrays["input_ids"].shape[1]
    packer.manifest["sequence_length"] = width
    assert packer.pack({"message": "中文"}, question)[0]["input_ids"].shape == (1, width)
    packer.manifest["sequence_length"] -= 1
    with pytest.raises(ValueError, match="no truncation"):
        packer.pack({"message": "中文"}, question)


async def test_coreai_invalid_logits_rejected():
    from types import SimpleNamespace

    import numpy as np

    engine = CoreAIVelaEngine.__new__(CoreAIVelaEngine)
    engine._model_name = "vllm-sr/vela3-fast"
    engine._packer = SimpleNamespace(
        manifest={"output": "logits"},
        pack=lambda state, q: ({"attention_mask": np.ones((1, 4))}, ["a", "b"]),
    )
    engine._session = SimpleNamespace(run=lambda arrays: {"logits": np.array([[np.nan, 1]])})
    with pytest.raises(FloatingPointError):
        await engine.infer(state="", questions={"q": {"type": "choice"}})
