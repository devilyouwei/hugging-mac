from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from hugging_mac_sdk import ArtifactFormat, ConverterRegistry, ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import (
    ResourceIntegrityError,
    ResourceNotFoundError,
    UnsupportedRuntimeError,
)
from hugging_mac_sdk.models.audio8_tts import AUDIO8_TTS_DEFINITION, register_audio8_tts
from hugging_mac_sdk.models.audio8_tts.coreai import _settled_thread
from hugging_mac_sdk.models.audio8_tts.definition import COREAI_CONVERTER, COREAI_LAYOUT
from hugging_mac_sdk.models.audio8_tts.utils.types import TtsEngineOutput

MODEL = AUDIO8_TTS_DEFINITION.manifest.model_id
VARIANT = "0.1b-preview"
TARGET = AUDIO8_TTS_DEFINITION.get_artifact("coreai", variant=VARIANT)
SOURCE = AUDIO8_TTS_DEFINITION.get_artifact("pytorch", variant=VARIANT)


def populate(artifact, root: Path) -> Path:
    path = artifact.resolve(root)
    for name in artifact.required_files:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps({"model_type": "arktts"}) if name.name == "config.json" else "data"
        )
    return path


def source_files(root: Path) -> None:
    populate(SOURCE, root)
    for share in AUDIO8_TTS_DEFINITION.required_shared_artifacts(
        runtime="pytorch", variant=VARIANT
    ):
        populate(share, root)


def test_coreai_registration_and_variant_coverage() -> None:
    registry, converters = ModelRegistry(), ConverterRegistry()
    definition = register_audio8_tts(registry, converters)
    assert converters.get(COREAI_CONVERTER.converter_id) is COREAI_CONVERTER
    instance = definition.create(runtime="coreai", variant=VARIANT)
    assert instance.info().runtime == "coreai"
    assert instance.supports(SpeechSynthesis)
    assert SOURCE.source.repo_id == "Edge0/Audio8-TTS-Preview-0.1b"
    assert TARGET.convert and TARGET.required_shares == SOURCE.required_shares
    with pytest.raises(ResourceNotFoundError):
        definition.get_artifacts(runtime="coreai", variant="0.6b-preview")


async def test_resources_complete_inventory_conversion_overwrite_and_delete(tmp_path, monkeypatch):
    source_files(tmp_path)
    sdk = ModelSdk()
    register_audio8_tts(sdk.registry)
    options = {"model_home": tmp_path, "runtime": "coreai"}
    before = await sdk.resources.status(MODEL, variant=VARIANT, options=options)
    assert not next(x for x in before.runtimes if x.runtime == "coreai").available

    def build(source, output):
        assert source == SOURCE.resolve(tmp_path)
        for name in TARGET.required_files:
            file = output / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("converted")

    monkeypatch.setattr(COREAI_CONVERTER, "_build", build)
    after = await sdk.resources.convert(
        MODEL, ArtifactFormat.COREAI, variant=VARIANT, options=options
    )
    assert next(x for x in after.runtimes if x.runtime == "coreai").available
    with pytest.raises(FileExistsError):
        await sdk.resources.convert(MODEL, ArtifactFormat.COREAI, variant=VARIANT, options=options)

    missing = TARGET.resolve(tmp_path) / TARGET.required_files[0]
    missing.unlink()
    incomplete = await sdk.resources.status(MODEL, variant=VARIANT, options=options)
    assert not next(x for x in incomplete.runtimes if x.runtime == "coreai").available
    await sdk.resources.convert(
        MODEL, ArtifactFormat.COREAI, variant=VARIANT, options=options, overwrite=True
    )

    def fail(source, output):
        raise RuntimeError("export failed")

    monkeypatch.setattr(COREAI_CONVERTER, "_build", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        await sdk.resources.convert(
            MODEL, ArtifactFormat.COREAI, variant=VARIANT, options=options, overwrite=True
        )
    assert missing.read_text() == "converted"
    assert not list(TARGET.resolve(tmp_path).parent.glob(".audio8-coreai-*"))
    for _ in range(2):
        await sdk.resources.delete(MODEL, variant=VARIANT, runtime="coreai", options=options)
    assert SOURCE.resolve(tmp_path).is_dir()
    assert not TARGET.resolve(tmp_path).exists()


async def test_missing_resources_fail_before_loading(tmp_path):
    instance = AUDIO8_TTS_DEFINITION.create(
        runtime="coreai", variant=VARIANT, options={"model_home": tmp_path}
    )
    with pytest.raises(ResourceNotFoundError):
        await instance._engine.resolve()
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    with pytest.raises(UnsupportedRuntimeError):
        await provider.convert("0.6b-preview", ArtifactFormat.COREAI, {"model_home": tmp_path})


async def test_coreai_capability_through_handle(tmp_path, monkeypatch):
    class Engine:
        runtime_name = "coreai"
        device = "auto"

        def __init__(self, *args):
            self.loaded = False

        async def resolve(self):
            return tmp_path

        async def load(self, path):
            self.loaded = True

        async def infer(self, request):
            assert self.loaded
            return TtsEngineOutput(np.zeros(2048, dtype="<f4").tobytes(), 44100, 2048 / 44100, 1)

        async def close(self):
            self.loaded = False

    monkeypatch.setattr(
        "hugging_mac_sdk.models.audio8_tts.definition.CoreAIAudio8TtsEngine", Engine
    )
    sdk = ModelSdk()
    register_audio8_tts(sdk.registry)
    for _ in range(2):
        async with await sdk.load(MODEL, variant=VARIANT, runtime="coreai") as handle:
            result = await handle.require(SpeechSynthesis).synthesize(
                SpeechSynthesisRequest(text="你好")
            )
            assert result.runtime == "coreai"
            assert result.sample_rate == 44100
            assert len(result.audio) == 8192


async def test_failed_coreai_load_closes_partial_sessions(tmp_path, monkeypatch):
    instance = AUDIO8_TTS_DEFINITION.create(
        runtime="coreai", variant=VARIANT, options={"model_home": tmp_path}
    )
    closed = []

    class Session:
        async def close(self):
            closed.append(True)

    calls = 0

    async def create(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("codec load failed")
        return Session()

    monkeypatch.setattr(
        "hugging_mac_sdk.models.audio8_tts.coreai.CoreAIProvider.create_session", create
    )
    with pytest.raises(RuntimeError, match="codec load failed"):
        await instance._engine.load(tmp_path)
    assert closed == [True]
    await instance._engine.close()
    assert closed == [True]


async def test_cancelled_native_call_settles_before_releasing_lock():
    entered, release, completed = Event(), Event(), Event()

    def work():
        entered.set()
        assert release.wait(5)
        completed.set()

    task = asyncio.create_task(_settled_thread(work))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed.is_set()


def test_fast_cache_resets_between_frames_and_empty_audio(tmp_path):
    torch = pytest.importorskip("torch")
    engine = AUDIO8_TTS_DEFINITION.create(
        runtime="coreai", variant=VARIANT, options={"model_home": tmp_path}
    )._engine
    seen = []

    def run(inputs):
        seen.append(inputs["keys"].copy())
        return {
            "logits": np.zeros((1, 4), dtype=np.float32),
            "new_keys": inputs["keys"] + 1,
            "new_values": inputs["values"] + 1,
        }

    engine._fast = SimpleNamespace(run=run)
    engine._codec = SimpleNamespace(run=lambda _: pytest.fail("Empty audio must not run the codec"))
    engine._cpu._torch = torch
    engine._cpu._model = SimpleNamespace(
        config=SimpleNamespace(
            n_fast_layer=2, fast_n_local_heads=1, num_codebooks=3, fast_head_dim=2
        )
    )
    engine._fast_step(torch.zeros(1, 1, 4), 0)
    engine._fast_step(torch.zeros(1, 1, 4), 1)
    engine._fast_step(torch.zeros(1, 1, 4), 0)
    assert [x.max() for x in seen] == [0, 1, 0]
    waveform, lengths = engine._decode_audio(torch.empty(1, 3, 0, dtype=torch.long))
    assert waveform.shape == (1, 0) and lengths.tolist() == [0]


@pytest.mark.smoke
async def test_managed_coreai_audio8_synthesis():
    pytest.importorskip("coreai")
    root = Path(__file__).resolve().parents[2] / "models"
    if not TARGET.resolve(root).is_dir():
        pytest.skip("Convert the managed Audio8 Core AI artifact first")
    sdk = ModelSdk()
    register_audio8_tts(sdk.registry)
    async with await sdk.load(
        MODEL, runtime="coreai", variant=VARIANT, options={"model_home": root, "device": "cpu"}
    ) as handle:
        result = await handle.require(SpeechSynthesis).synthesize(
            SpeechSynthesisRequest(text="你好, 这是一个测试。", max_new_tokens=48, do_sample=False)
        )
        audio = np.frombuffer(result.audio, dtype="<f4")
        assert result.runtime == "coreai" and result.sample_rate == 44100
        assert len(audio) > 0 and np.isfinite(audio).all() and np.abs(audio).max() > 0


async def test_coreai_download_and_unsafe_delete(tmp_path, monkeypatch):
    downloads = []

    async def download(self, source, destination, **kwargs):
        downloads.append((source.repo_id, kwargs["overwrite"]))
        artifact = (
            TARGET
            if destination == TARGET.resolve(tmp_path)
            else next(
                a
                for a in AUDIO8_TTS_DEFINITION.shared_artifacts
                if a.resolve(tmp_path) == destination
            )
        )
        populate(artifact, tmp_path)

    monkeypatch.setattr(
        "hugging_mac_sdk.resources.downloader.ResourceDownloader.download", download
    )
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    options = {"model_home": tmp_path, "runtime": "coreai"}
    await provider.download_source(VARIANT, options)
    assert len(downloads) == 2
    assert all(repo == "hugging-mac/audio8-tts-0.1b-coreai" for repo, _ in downloads)
    assert not SOURCE.resolve(tmp_path).exists()
    status = await provider.status(VARIANT, options)
    assert next(x for x in status.runtimes if x.runtime == "coreai").available
    await provider.download_source(VARIANT, options)
    assert len(downloads) == 2
    await provider.download_source(VARIANT, options, overwrite=True)
    assert len(downloads) == 4 and all(overwrite for _, overwrite in downloads[-2:])
    await provider.delete(VARIANT, options, runtime="coreai")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("keep")
    target = TARGET.resolve(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        await provider.delete(VARIANT, options, runtime="coreai")
    assert marker.read_text() == "keep"


async def test_coreai_download_rejects_incomplete_remote_inventory(tmp_path, monkeypatch):
    async def download(self, source, destination, **kwargs):
        path = populate(TARGET, tmp_path)
        (path / TARGET.required_files[-1]).unlink()

    monkeypatch.setattr(
        "hugging_mac_sdk.resources.downloader.ResourceDownloader.download", download
    )
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    options = {"model_home": tmp_path, "runtime": "coreai"}
    with pytest.raises(ResourceIntegrityError, match="incomplete"):
        await provider.download_source(VARIANT, options)
    status = await provider.status(VARIANT, options)
    assert not next(x for x in status.runtimes if x.runtime == "coreai").available
    with pytest.raises(UnsupportedRuntimeError, match="no Core AI download"):
        await provider.download_source("0.6b-preview", options)


@pytest.mark.smoke
async def test_managed_coreai_graphs_match_pytorch():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("coreai")
    from hugging_mac_sdk.models.audio8_tts.torch import (
        _install_falcon_h1_cache_compatibility,
        _restore_falcon_h1_buffers,
        _restore_rope_buffers,
    )
    from hugging_mac_sdk.runtime.coreai import CoreAIProvider

    root = Path(__file__).resolve().parents[2] / "models"
    path = TARGET.resolve(root)
    if not path.is_dir() or not SOURCE.resolve(root).is_dir():
        pytest.skip("Convert the managed Audio8 Core AI artifact first")
    _install_falcon_h1_cache_compatibility(transformers)
    model = transformers.AutoModel.from_pretrained(
        SOURCE.resolve(root), local_files_only=True, trust_remote_code=True, dtype=torch.float32
    ).eval()
    _restore_rope_buffers(torch, model)
    _restore_falcon_h1_buffers(transformers, model)
    config = model.config
    model._setup_generation_caches(1, config.num_codebooks, torch.float32)
    shape = (
        config.n_fast_layer,
        1,
        config.fast_n_local_heads,
        config.num_codebooks,
        config.fast_head_dim,
    )
    keys, values = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    fast = await CoreAIProvider().create_session(
        path / COREAI_LAYOUT.fast_graph, device="cpu", options={}
    )
    try:
        generator = torch.Generator().manual_seed(17)
        for position in range(config.num_codebooks):
            hidden = torch.randn(1, 1, config.fast_dim, generator=generator)
            with torch.inference_mode():
                expected = model._fast_step(hidden, position).numpy()
            actual = fast.run(
                {
                    "hidden": hidden.numpy(),
                    "position": np.asarray([position], np.int32),
                    "keys": keys,
                    "values": values,
                }
            )
            keys, values = actual["new_keys"], actual["new_values"]
            np.testing.assert_allclose(actual["logits"], expected, rtol=1e-3, atol=1e-3)
    finally:
        await fast.close()
    codec = model.load_codec(device="cpu", dtype=torch.float32)
    decoder = await CoreAIProvider().create_session(
        path / COREAI_LAYOUT.codec_graph, device="cpu", options={}
    )
    try:
        for frames in (1, 8, 17):
            codes = torch.randint(0, 1024, (1, config.num_codebooks, frames), generator=generator)
            with torch.inference_mode():
                expected = codec.decode(codes).numpy()
            actual = decoder.run({"codes": codes.numpy().astype(np.int32)})
            np.testing.assert_allclose(actual["waveform"], expected, rtol=1e-3, atol=1e-3)
    finally:
        await decoder.close()


async def test_cancelled_conversion_does_not_publish(tmp_path, monkeypatch):
    source_files(tmp_path)
    output = populate(TARGET, tmp_path)
    entered, release = Event(), Event()

    def build(source, staged):
        entered.set()
        assert release.wait(5)
        for name in TARGET.required_files:
            file = staged / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("replacement")

    monkeypatch.setattr(COREAI_CONVERTER, "_build", build)
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    task = asyncio.create_task(
        provider.convert(VARIANT, ArtifactFormat.COREAI, {"model_home": tmp_path}, overwrite=True)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (output / TARGET.required_files[0]).read_text() == "data"
    assert not list(output.parent.glob(".audio8-coreai-*"))
