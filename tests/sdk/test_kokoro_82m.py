from __future__ import annotations

from pathlib import Path

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
from hugging_mac_sdk.models.kokoro_82m.utils.types import KokoroEngineOutput
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest
from hugging_mac_sdk.schemas.transcription import AudioInput


class FakeKokoroEngine:
    runtime_name = "pytorch-mps"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False

    @property
    def device(self) -> str:
        return "mps"

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


def test_kokoro_manifest_and_registration() -> None:
    assert KOKORO_82M_MANIFEST.default_variant == "v1.0"
    assert KOKORO_82M_MANIFEST.default_runtime == "pytorch-mps"
    assert KOKORO_82M_MANIFEST.capabilities == {"speech-synthesis"}
    assert KOKORO_82M_MANIFEST.license == "Apache-2.0"
    assert {runtime.name for runtime in KOKORO_82M_MANIFEST.runtimes} == {"pytorch-mps"}

    models = ModelRegistry()
    definition = register_kokoro_82m(models)
    pytorch = definition.create(runtime="pytorch-mps", variant="v1.0")

    assert isinstance(pytorch, Kokoro82mInstance)
    assert pytorch.supports(SpeechSynthesis)  # type: ignore[type-abstract]


async def test_kokoro_instance_orchestrates_synthesis(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeKokoroEngine(artifact)
    config = Kokoro82mInstanceConfig(runtime="pytorch-mps")
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
    assert response.runtime == "pytorch-mps"
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
