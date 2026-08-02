from __future__ import annotations

import sys
from pathlib import Path
from threading import Event
from types import ModuleType

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.models.kokoro_82m import (
    KOKORO_82M_MANIFEST,
    register_kokoro_82m,
)
from hugging_mac_sdk.models.kokoro_82m.config import Kokoro82mInstanceConfig
from hugging_mac_sdk.models.kokoro_82m.instance import Kokoro82mInstance
from hugging_mac_sdk.models.kokoro_82m.mlx import MlxKokoro82mEngine
from hugging_mac_sdk.models.kokoro_82m.utils.types import KokoroEngineOutput
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput


class FakeKokoroEngine:
    runtime_name = "mlx"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    @property
    def device(self) -> str:
        return "gpu"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(self, request: SpeechSynthesisRequest) -> KokoroEngineOutput:
        assert request.voice == "af_heart"
        assert request.speed == 1.1
        return KokoroEngineOutput(
            audio=b"\x00\x00\x00\x00" * 24000,
            sample_rate=24000,
            duration_seconds=1.0,
            generated_tokens=40,
        )

    async def close(self) -> None:
        self.closed = True


class FakeMlxResult:
    audio = np.zeros(24000, dtype=np.float32)
    sample_rate = 24000
    token_count = 40


class FakeMlxModel:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    def generate(self, **kwargs: object):
        assert kwargs == {
            "text": "Hello from Kokoro.",
            "voice": str(self.artifact / "voices" / "af_heart.safetensors"),
            "speed": 1.1,
            "lang_code": "a",
        }
        yield FakeMlxResult()


def test_kokoro_manifest_and_registration() -> None:
    assert KOKORO_82M_MANIFEST.default_variant == "bf16"
    assert KOKORO_82M_MANIFEST.default_runtime == "mlx"
    assert KOKORO_82M_MANIFEST.capabilities == {"speech-synthesis"}
    assert KOKORO_82M_MANIFEST.license == "Apache-2.0"
    assert {runtime.name for runtime in KOKORO_82M_MANIFEST.runtimes} == {"mlx"}

    models = ModelRegistry()
    definition = register_kokoro_82m(models)
    mlx = definition.create(runtime="mlx", variant="bf16")

    assert isinstance(mlx, Kokoro82mInstance)
    assert mlx.supports(SpeechSynthesis)  # type: ignore[type-abstract]


async def test_kokoro_instance_orchestrates_synthesis(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeKokoroEngine(artifact)
    config = Kokoro82mInstanceConfig(runtime="mlx")
    instance = Kokoro82mInstance(config, engine)

    await instance.load()
    synthesizer = instance.require(SpeechSynthesis)  # type: ignore[type-abstract]
    response = await synthesizer.synthesize(
        SpeechSynthesisRequest(
            text="Hello from Kokoro.",
            voice="af_heart",
            language="a",
            speed=1.1,
        )
    )

    assert instance.state is ModelState.READY
    assert response.sample_rate == 24000
    assert response.runtime == "mlx"
    assert response.duration_seconds == 1.0
    assert response.generated_tokens == 40

    await instance.unload()
    assert engine.closed


async def test_kokoro_rejects_reference_voice_cloning(tmp_path: Path) -> None:
    engine = FakeKokoroEngine(tmp_path)
    instance = Kokoro82mInstance(Kokoro82mInstanceConfig(), engine)
    await instance.load()

    with pytest.raises(InferenceError, match="does not support reference cloning"):
        await instance.synthesize(
            SpeechSynthesisRequest(
                text="hello",
                reference_audio=AudioInput(data=b"audio"),
                reference_text="reference",
            )
        )


def test_kokoro_mlx_engine_maps_request_and_audio(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "af_heart.safetensors").touch()
    config = Kokoro82mInstanceConfig(source_path=tmp_path)
    engine = MlxKokoro82mEngine(config, object())  # type: ignore[arg-type]
    engine._model = FakeMlxModel(tmp_path)
    engine._artifact = tmp_path

    output = engine._infer_sync(
        SpeechSynthesisRequest(
            text="Hello from Kokoro.",
            voice="af_heart",
            language="a",
            speed=1.1,
        ),
        Event(),
    )

    assert output.sample_rate == 24000
    assert output.duration_seconds == 1.0
    assert output.generated_tokens == 40
    assert len(output.audio) == 24000 * 4


async def test_kokoro_mlx_engine_explicitly_selects_legacy_model_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    loaded_model = object()
    utils = ModuleType("mlx_audio.tts.utils")

    def fake_load(path: str, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return loaded_model

    utils.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    engine = MlxKokoro82mEngine(
        Kokoro82mInstanceConfig(source_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )

    await engine.load(tmp_path)

    assert engine._model is loaded_model
    assert engine._artifact == tmp_path
    assert calls == [(str(tmp_path), {"model_type": "kokoro"})]


def test_kokoro_mlx_engine_rejects_missing_or_unsafe_voice(tmp_path: Path) -> None:
    engine = MlxKokoro82mEngine(
        Kokoro82mInstanceConfig(source_path=tmp_path),
        object(),  # type: ignore[arg-type]
    )
    engine._artifact = tmp_path

    with pytest.raises(FileNotFoundError, match="not downloaded"):
        engine._voice_path("af_missing")
    with pytest.raises(ValueError, match="packaged voice name"):
        engine._voice_path("../af_heart")
