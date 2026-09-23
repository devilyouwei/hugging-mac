from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.audio8_tts import (
    AUDIO8_TTS_DEFINITION,
    AUDIO8_TTS_MANIFEST,
    register_audio8_tts,
)
from hugging_mac_sdk.models.audio8_tts.config import Audio8TtsInstanceConfig
from hugging_mac_sdk.models.audio8_tts.instance import Audio8TtsInstance
from hugging_mac_sdk.models.audio8_tts.torch import (
    TorchAudio8TtsEngine,
    _install_falcon_h1_cache_compatibility,
    _restore_falcon_h1_buffers,
    _restore_rope_buffers,
)
from hugging_mac_sdk.models.audio8_tts.utils.types import TtsEngineOutput
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest


class FakeTtsEngine:
    runtime_name = "pytorch"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    @property
    def device(self) -> str:
        return "cpu"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        assert request.text == "你好, Audio8."
        return TtsEngineOutput(
            audio=b"\x00\x00\x00\x00" * 44100,
            sample_rate=44100,
            duration_seconds=1.0,
            generated_tokens=22,
        )

    async def close(self) -> None:
        self.closed = True


def test_audio8_tts_artifact_declares_its_source() -> None:
    assert AUDIO8_TTS_MANIFEST.default_variant == "0.1b-preview"
    assert Audio8TtsInstanceConfig().variant == "0.1b-preview"
    assert AUDIO8_TTS_MANIFEST.capabilities == {"speech-synthesis"}
    assert AUDIO8_TTS_MANIFEST.license == "audio8-community-license-v1.0"
    runtime = AUDIO8_TTS_MANIFEST.runtimes[0]
    assert runtime.name == "pytorch"
    assert runtime.devices == ("cpu", "mps")
    assert runtime.dtypes == ("float32",)
    source = AUDIO8_TTS_DEFINITION.get_artifact("pytorch", "source", "0.6b-preview").source
    assert source is not None
    assert source.revision == "main"  # type: ignore[union-attr]
    assert source.allow_patterns == (  # type: ignore[union-attr]
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "*.py",
        "*.pth",
        "*.safetensors",
    )
    shared = {artifact.artifact_id: artifact for artifact in AUDIO8_TTS_DEFINITION.shared_artifacts}
    assert set(shared) == {"tokenizer-0.1b", "tokenizer-0.6b"}
    for tokenizer in shared.values():
        prefix = "_shared/tokenizer-0.1b/" if tokenizer.artifact_id == "tokenizer-0.1b" else ""
        assert tokenizer.source.allow_patterns == (  # type: ignore[union-attr]
            prefix + "special_tokens_map.json",
            prefix + "tokenizer.json",
            prefix + "tokenizer_config.json",
        )


def test_registered_factory_exposes_speech_synthesis() -> None:
    registry = ModelRegistry()
    definition = register_audio8_tts(registry)
    instance = definition.create(runtime="pytorch", variant="0.6b-preview")

    assert isinstance(instance, Audio8TtsInstance)
    assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


def test_audio8_tts_manifest_contains_0_1b_variant() -> None:
    variant = AUDIO8_TTS_MANIFEST.get_variant("0.1b-preview")
    source = AUDIO8_TTS_DEFINITION.get_artifact("pytorch", "source", "0.1b-preview").source
    assert source is not None

    assert variant.metadata["parameters"] == 169779904
    assert variant.metadata["license"] == "audio8-community-license-v1.0"
    assert source.repo_id == "Edge0/Audio8-TTS-Preview-0.1b"  # type: ignore[union-attr]
    assert source.revision == "main"  # type: ignore[union-attr]

    instance = AUDIO8_TTS_DEFINITION.create(runtime="pytorch", variant="0.1b-preview")
    assert isinstance(instance, Audio8TtsInstance)

    source_artifact = AUDIO8_TTS_DEFINITION.get_artifact("pytorch", "source", "0.1b-preview")
    tokenizer_artifact = AUDIO8_TTS_DEFINITION.required_shared_artifacts(
        runtime="pytorch", variant="0.1b-preview", artifact_id="source"
    )
    assert source_artifact.required_shares == ("tokenizer-0.1b",)
    assert len(tokenizer_artifact) == 1
    assert tokenizer_artifact[0].artifact_id == "tokenizer-0.1b"
    tokenizer_source = tokenizer_artifact[0].source
    assert tokenizer_source.repo_id == "hugging-mac/audio8-tts-0.1b-coreai"  # type: ignore[union-attr]
    assert tokenizer_source.strip_prefix == Path("_shared/tokenizer-0.1b")  # type: ignore[union-attr]


async def test_audio8_tts_0_1b_status_exposes_pytorch(tmp_path: Path) -> None:
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    assert provider is not None

    status = await provider.status("0.1b-preview", {"model_home": tmp_path})

    assert {item.runtime for item in status.runtimes} == {"pytorch", "coreai"}
    assert status.revision == "main"


async def test_audio8_tts_resource_status_exposes_both_runtimes(tmp_path: Path) -> None:
    provider = AUDIO8_TTS_DEFINITION.resource_provider
    assert provider is not None

    status = await provider.status("0.6b-preview", {"model_home": tmp_path})

    assert {item.runtime for item in status.runtimes} == {"pytorch", "mlx"}


def test_synthesis_request_rejects_reference_text_without_audio() -> None:
    with pytest.raises(ValueError, match="requires reference_audio"):
        SpeechSynthesisRequest(text="hello", reference_text="reference")


def test_audio8_tts_0_1b_pytorch_defaults_to_cpu() -> None:
    instance = AUDIO8_TTS_DEFINITION.create(runtime="pytorch", variant="0.1b-preview")

    assert instance.info().runtime == "pytorch"
    assert instance.info().device == "cpu"


def test_audio8_tts_0_6b_pytorch_defaults_to_cpu() -> None:
    instance = AUDIO8_TTS_DEFINITION.create(runtime="pytorch", variant="0.6b-preview")
    assert instance.info().device == "cpu"

    mps = AUDIO8_TTS_DEFINITION.create(
        runtime="pytorch", variant="0.6b-preview", options={"device": "mps"}
    )
    assert mps.info().device == "mps"


def test_audio8_tts_uses_cpu_by_default() -> None:
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(),
        resources=object(),  # type: ignore[arg-type]
    )

    assert engine.device == "cpu"

    class FakeTorch:
        float32 = "fp32"

    assert engine._resolve_dtype(FakeTorch) == "fp32"


async def test_audio8_tts_falls_back_when_mps_inference_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(device="mps"),
        resources=object(),  # type: ignore[arg-type]
    )
    engine._model = object()
    engine._artifact = Path("model")
    attempts = 0

    def fake_infer(_: SpeechSynthesisRequest) -> TtsEngineOutput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("MPS operator failed")
        return TtsEngineOutput(
            audio=b"\x00\x00\x00\x00",
            sample_rate=44100,
            duration_seconds=1 / 44100,
            generated_tokens=1,
        )

    def fake_reload() -> None:
        engine._device = "cpu"

    monkeypatch.setattr(engine, "_infer_sync", fake_infer)
    monkeypatch.setattr(engine, "_reload_on_cpu_sync", fake_reload)

    output = await engine.infer(SpeechSynthesisRequest(text="hello"))

    assert attempts == 2
    assert engine.device == "cpu"
    assert output.generated_tokens == 1


async def test_audio8_tts_falls_back_when_mps_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hugging_mac_sdk.models.audio8_tts.torch as torch_module

    class FakeProvider:
        def is_available(self) -> bool:
            return True

        def resolve_device(self, requested: str, *, allow_cpu_fallback: bool) -> str:
            assert requested == "mps"
            assert allow_cpu_fallback
            return "mps"

    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(device="mps"),
        resources=object(),  # type: ignore[arg-type]
    )
    attempts: list[str] = []

    def fake_load(_: Path) -> None:
        attempts.append(engine.device)
        if engine.device == "mps":
            raise RuntimeError("MPS load failed")

    monkeypatch.setattr(torch_module, "TorchProvider", FakeProvider)
    monkeypatch.setattr(engine, "_load_sync", fake_load)

    await engine.load(tmp_path)

    assert attempts == ["mps", "cpu"]
    assert engine.device == "cpu"


async def test_pytorch_engine_keeps_merged_model_view_until_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hugging_mac_sdk.models.audio8_tts.torch as torch_module

    loaded_paths: list[Path] = []

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

    class FakeLoader:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> object:
            loaded_paths.append(Path(path))
            return FakeModel()

    fake_torch = SimpleNamespace(float32="float32")
    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeLoader,
        AutoModel=FakeLoader,
    )
    monkeypatch.setattr(
        torch_module.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )
    monkeypatch.setattr(torch_module, "_restore_rope_buffers", lambda *_: None)

    tokenizer_path = tmp_path / "tokenizer"
    artifact_path = tmp_path / "model"
    tokenizer_path.mkdir()
    artifact_path.mkdir()
    resources = SimpleNamespace(tokenizer_path=tokenizer_path)
    engine = TorchAudio8TtsEngine(
        Audio8TtsInstanceConfig(device="cpu"),
        resources,  # type: ignore[arg-type]
    )

    engine._load_sync(artifact_path)

    assert engine._model is not None
    assert loaded_paths[0].is_dir()

    await engine.close()

    assert not loaded_paths[0].exists()


def test_audio8_tts_restores_transformers_v5_rope_buffers() -> None:
    torch = pytest.importorskip("torch")

    class FakeConfig:
        max_seq_len = 32
        head_dim = 8
        rope_base = 1_000_000.0
        num_codebooks = 10
        fast_head_dim = 8

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = FakeConfig()
            self.register_buffer("freqs_cis", torch.full((32, 4, 2), torch.nan))
            self.register_buffer("fast_freqs_cis", torch.full((10, 4, 2), torch.nan))

    model = FakeModel()
    _restore_rope_buffers(torch, model)

    assert model.freqs_cis.shape == (32, 4, 2)
    assert model.fast_freqs_cis.shape == (10, 4, 2)
    assert model.freqs_cis.dtype == torch.bfloat16
    assert torch.isfinite(model.freqs_cis).all()
    assert model.freqs_cis[0, :, 0].tolist() == [1.0] * 4
    assert model.freqs_cis[0, :, 1].tolist() == [0.0] * 4


def test_audio8_tts_restores_transformers_v5_falcon_mup_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hugging_mac_sdk.models.audio8_tts.torch as torch_module

    torch = pytest.importorskip("torch")
    layers = [
        SimpleNamespace(mamba=SimpleNamespace(mup_vector=torch.zeros(1, 1, 4))) for _ in range(2)
    ]
    model = SimpleNamespace(slow=SimpleNamespace(config=object(), layers=layers))
    fake_falcon_h1 = SimpleNamespace(
        compute_mup_vector=lambda _: torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    )
    monkeypatch.setattr(torch_module.importlib, "import_module", lambda _: fake_falcon_h1)

    _restore_falcon_h1_buffers(SimpleNamespace(__version__="5.14.1"), model)

    assert all(layer.mamba.mup_vector.tolist() == [[[1.0, 2.0, 3.0, 4.0]]] for layer in layers)


def test_audio8_tts_adapts_removed_transformers_v4_falcon_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hugging_mac_sdk.models.audio8_tts.torch as torch_module

    class FakeDynamicCache:
        def __init__(self, *, config: object) -> None:
            self.config = config

    fake_falcon_h1 = SimpleNamespace(__name__="transformers.models.falcon_h1.modeling_falcon_h1")
    fake_transformers = SimpleNamespace(__version__="5.14.1", DynamicCache=FakeDynamicCache)
    monkeypatch.setattr(torch_module.importlib, "import_module", lambda _: fake_falcon_h1)

    _install_falcon_h1_cache_compatibility(fake_transformers)

    compatibility_cache = fake_falcon_h1.FalconHybridMambaAttentionDynamicCache(
        "falcon-config",
        1,
        "float32",
        devices=["cpu"],
    )
    assert isinstance(compatibility_cache, FakeDynamicCache)
    assert compatibility_cache.config == "falcon-config"


def test_audio8_tts_keeps_existing_falcon_cache_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hugging_mac_sdk.models.audio8_tts.torch as torch_module

    existing_cache = object()
    fake_falcon_h1 = SimpleNamespace(
        FalconHybridMambaAttentionDynamicCache=existing_cache,
    )
    fake_transformers = SimpleNamespace(__version__="5.14.1", DynamicCache=object)
    monkeypatch.setattr(torch_module.importlib, "import_module", lambda _: fake_falcon_h1)

    _install_falcon_h1_cache_compatibility(fake_transformers)

    assert fake_falcon_h1.FalconHybridMambaAttentionDynamicCache is existing_cache


async def test_audio8_tts_instance_orchestrates_synthesis(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeTtsEngine(artifact)
    instance = Audio8TtsInstance(Audio8TtsInstanceConfig(), engine, AUDIO8_TTS_MANIFEST)

    await instance.load()
    synthesizer = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
    response = await synthesizer.synthesize(SpeechSynthesisRequest(text="你好, Audio8."))

    assert instance.state is ModelState.READY
    assert engine.loaded
    assert response.sample_rate == 44100
    assert response.duration_seconds == 1.0
    assert response.generated_tokens == 22
    assert response.audio_format == "f32le"

    await instance.unload()
    assert engine.closed
