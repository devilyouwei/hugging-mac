"""Laya registration, checkpoint separation and typed action contracts."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hugging_mac_sdk import ModelSdk, ReusePolicy
from hugging_mac_sdk.capabilities import DecisionMaking
from hugging_mac_sdk.models.laya import LAYA_DEFINITION, register_laya
from hugging_mac_sdk.models.laya.coreai import CoreAILayaEngine
from hugging_mac_sdk.models.laya.definition import LAYA_CONFIG
from hugging_mac_sdk.schemas.decision import DecisionRequest


def test_lazy_registration() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from hugging_mac_sdk.models.laya import register_laya
from hugging_mac_sdk import ModelSdk
register_laya(ModelSdk().registry)
assert not {'torch','laya','numpy','tokenizers','coreai'} & sys.modules.keys()
""",
        ],
        check=True,
    )


def test_checkpoint_and_tokenizer_separation() -> None:
    source = LAYA_CONFIG.get_artifact(shared=False, variant="multilingual", runtime="pytorch")
    exported = LAYA_CONFIG.get_artifact(
        shared=False, variant="multilingual-legacy-s512", runtime="coreai"
    )
    assert source.source.repo_id == "convaiinnovations/laya-multilingual"
    assert exported.source.repo_id == "mlboydaisuke/Laya-Multilingual-CoreAI"
    assert source.required_shares != exported.required_shares
    assert exported.source.strip_prefix == Path("macos/wfp16-s512")
    with pytest.raises(ValueError, match="window"):
        LAYA_DEFINITION.runtime_factories["coreai"]({"max_length": 8192})


async def test_typed_actions_survive_sdk_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(CoreAILayaEngine, "resolve", AsyncMock(return_value=Path("bundle")))
    monkeypatch.setattr(CoreAILayaEngine, "load", AsyncMock())
    monkeypatch.setattr(CoreAILayaEngine, "close", AsyncMock())
    monkeypatch.setattr(
        CoreAILayaEngine,
        "infer",
        AsyncMock(
            return_value={
                "answers": {
                    "q": {
                        "type": "noul",
                        "noul": 0.8,
                        "confidence": 0.8,
                        "action": {"act_probability": 0.7},
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 0},
            }
        ),
    )
    sdk = ModelSdk()
    register_laya(sdk.registry)
    async with await sdk.acquire(
        "convaiinnovations/laya", runtime="coreai", reuse=ReusePolicy.DEDICATED
    ) as handle:
        result = await handle.require(DecisionMaking).decide(
            DecisionRequest(
                state="hello", questions={"q": {"type": "noul", "instructions": "Positive?"}}
            )
        )
        assert result.answers["q"].action.act_probability == 0.7
        assert result.answers["q"].confidence == 0.8


async def test_managed_tokenizer_view_and_inventory(tmp_path) -> None:
    from hugging_mac_sdk.models.laya.config import LayaInstanceConfig
    from hugging_mac_sdk.models.laya.resources import LayaResourceResolver

    for variant, runtime in (("multilingual", "pytorch"), ("multilingual-legacy-s512", "coreai")):
        artifact = LAYA_CONFIG.get_artifact(shared=False, variant=variant, runtime=runtime)
        for item in LAYA_CONFIG.get_artifacts(variant=variant, include_shared=True):
            for required in item.required_files:
                path = item.resolve(tmp_path) / required
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
        resolver = LayaResourceResolver(
            LayaInstanceConfig(model_home=tmp_path, variant=variant, runtime=runtime),
            manifest=LAYA_CONFIG.manifest,
            artifact=artifact,
            shared_artifacts=LAYA_DEFINITION.shared_artifacts,
        )
        assert await resolver.resolve() == artifact.resolve(tmp_path)
        with resolver.view() as view:
            assert (view / "tokenizer/tokenizer.json").read_text() == "fixture"
        status = await LAYA_DEFINITION.resource_provider.status(variant, {"model_home": tmp_path})
        assert all(item.available for item in status.artifacts)


@pytest.mark.parametrize(
    "variant,runtime",
    [
        ("multilingual", "pytorch"),
        ("multilingual-legacy-s256", "coreai"),
        ("multilingual-legacy-s512", "coreai"),
    ],
)
def test_all_declared_factories(variant, runtime):
    instance = LAYA_DEFINITION.runtime_factories[runtime]({"variant": variant})
    assert instance.info().variant == variant
    assert set(instance.capabilities) == {DecisionMaking}


async def test_coreai_partial_load_closes_main_session(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from hugging_mac_sdk.models.laya.config import LayaInstanceConfig
    from hugging_mac_sdk.models.laya.utils import packing

    main = SimpleNamespace(close=AsyncMock())
    create = AsyncMock(side_effect=[main, RuntimeError("act failed")])
    monkeypatch.setattr("hugging_mac_sdk.models.laya.coreai.CoreAIProvider.create_session", create)
    monkeypatch.setattr(
        packing,
        "LayaPacker",
        lambda root: SimpleNamespace(
            meta={"assets": {"main": "test.aimodel"}},
            layout={"functions": {"main": "encoder", "act": "action"}},
        ),
    )
    engine = CoreAILayaEngine(
        LayaInstanceConfig(), SimpleNamespace(view=lambda: nullcontext(tmp_path)), model_name="test"
    )
    with pytest.raises(RuntimeError, match="act failed"):
        await engine.load(tmp_path)
    assert create.call_args_list[0].kwargs["options"] == {"function_name": "encoder"}
    assert create.call_args_list[1].kwargs["options"] == {"function_name": "action"}
    main.close.assert_awaited_once()
    await engine.close()
    main.close.assert_awaited_once()


@pytest.mark.parametrize(
    "variant,runtime",
    [
        ("multilingual", "pytorch"),
        ("multilingual-legacy-s256", "coreai"),
        ("multilingual-legacy-s512", "coreai"),
    ],
)
async def test_scoped_resource_download_overwrite_and_delete(
    tmp_path, monkeypatch, variant, runtime
):
    from hugging_mac_sdk.errors import ResourceIntegrityError

    sdk = ModelSdk()
    register_laya(sdk.registry)
    selected = LAYA_CONFIG.get_artifacts(variant=variant, runtime=runtime, include_shared=True)
    calls = []

    async def download(self, declaration, destination, **kwargs):
        artifact = next(a for a in selected if a.source == declaration)
        assert destination == artifact.resolve(tmp_path)
        calls.append((artifact.artifact_id, kwargs.get("overwrite")))
        for name in artifact.required_files:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")

    monkeypatch.setattr(
        "hugging_mac_sdk.resources.downloader.ResourceDownloader.download", download
    )
    options = {"model_home": tmp_path, "runtime": runtime}
    status = await sdk.resources.download_source(
        "convaiinnovations/laya", variant=variant, options=options
    )
    assert all(a.available for a in status.artifacts)
    first = len(calls)
    await sdk.resources.download_source("convaiinnovations/laya", variant=variant, options=options)
    assert len(calls) == first
    artifact = next(a for a in selected if not a.shared)
    (artifact.resolve(tmp_path) / artifact.required_files[-1]).unlink()
    status = await LAYA_DEFINITION.resource_provider.status(variant, options)
    assert not next(a for a in status.artifacts if not a.shared).available
    await sdk.resources.download_source(
        "convaiinnovations/laya", variant=variant, options=options, overwrite=True
    )
    assert calls[-1][1] is True
    provider = LAYA_DEFINITION.resource_provider
    with pytest.raises(ResourceIntegrityError):
        await provider.delete(variant, options | {"artifact_path": tmp_path / "external"})
    for _ in range(2):
        status = await provider.delete(variant, options)
        assert not next(a for a in status.artifacts if not a.shared).available
        assert all(a.available for a in status.artifacts if a.shared)
