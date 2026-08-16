from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import VoiceActivityDetection
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.silero import SILERO_MANIFEST, register_silero
from hugging_mac_sdk.models.silero.config import SILERO_SHA256, SileroInstanceConfig
from hugging_mac_sdk.models.silero.coreml import CoreMlSileroEngine
from hugging_mac_sdk.models.silero.instance import SileroInstance
from hugging_mac_sdk.models.silero.resources import SileroResourceResolver
from hugging_mac_sdk.models.silero.utils.postprocess import probabilities_to_segments
from hugging_mac_sdk.models.silero.utils.types import PreparedAudio, SileroInferenceOptions
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, UrlFileSource
from hugging_mac_sdk.schemas.transcription import AudioInput
from hugging_mac_sdk.schemas.voice_activity import VoiceActivityRequest


class FakeSileroEngine:
    runtime_name = "onnx"
    device = "CPUExecutionProvider"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.closed = False

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact

    async def probabilities(self, prepared: PreparedAudio) -> tuple[float, ...]:
        del prepared
        return (0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0)

    async def close(self) -> None:
        self.closed = True


def test_manifest_and_registration() -> None:
    assert SILERO_MANIFEST.capabilities == {"voice-activity-detection"}
    assert SILERO_MANIFEST.default_runtime == "coreml"
    assert {runtime.name for runtime in SILERO_MANIFEST.runtimes} == {"coreml", "onnx"}
    source = SILERO_MANIFEST.get_variant().resources[0]
    assert source.expected_sha256 == SILERO_SHA256  # type: ignore[union-attr]

    definition = register_silero(ModelRegistry())
    instance = definition.create(runtime="onnx", options={"device": "cpu"})

    assert isinstance(instance, SileroInstance)
    assert instance.supports(VoiceActivityDetection)  # type: ignore[type-abstract]


async def test_instance_detects_voice_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = PreparedAudio(np.zeros(3584, dtype=np.float32), 16000, 0.224)
    monkeypatch.setattr(
        "hugging_mac_sdk.models.silero.instance.prepare_audio",
        lambda *_args, **_kwargs: prepared,
    )
    engine = FakeSileroEngine(tmp_path / "silero.onnx")
    instance = SileroInstance(SileroInstanceConfig(runtime="onnx", device="cpu"), engine)
    await instance.load()

    detector = instance.require(VoiceActivityDetection)  # type: ignore[type-abstract]
    response = await detector.detect_voice_activity(
        VoiceActivityRequest(
            audio=AudioInput(data=b"audio"),
            min_speech_ms=30,
            min_silence_ms=30,
            speech_pad_ms=0,
        )
    )

    assert [(item.start_sample, item.end_sample) for item in response.segments] == [
        (512, 2048)
    ]
    assert response.sample_rate == 16000
    assert response.speech_seconds == pytest.approx(0.096)
    await instance.unload()
    assert engine.closed


def test_postprocess_uses_hysteresis_and_padding() -> None:
    segments = probabilities_to_segments(
        (0.1, 0.7, 0.4, 0.3, 0.2, 0.8, 0.1, 0.1),
        total_samples=4096,
        sample_rate=16000,
        chunk_samples=512,
        options=SileroInferenceOptions(0.5, 30, 30, 10, None),
    )

    assert [(item.start, item.end) for item in segments] == [(352, 1696), (2400, 3232)]


async def test_coreml_resolver_accepts_models_page_snapshot_layout(tmp_path: Path) -> None:
    artifact = tmp_path / "silero_vad.mlmodelc"
    nested = artifact / "silero_vad.mlmodelc"
    for name in ("coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"):
        target = nested / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
    sources = SILERO_MANIFEST.get_variant().resources
    resolver = SileroResourceResolver(
        UrlFileSource.model_validate(sources[0]),
        HuggingFaceSource.model_validate(sources[1]),
        SileroInstanceConfig(runtime="coreml", artifact_path=artifact),
    )

    resolved = await resolver.resolve_coreml()

    assert resolved.path == nested


async def test_coreml_engine_passes_explicit_streaming_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, np.ndarray]] = []

    class FakeSession:
        device = "all"

        def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls.append(inputs)
            value = np.asarray([0.2 + len(calls) / 10], dtype=np.float32)
            return {
                "probability": value,
                "h_out": np.ones((1, 1, 128), dtype=np.float32),
                "c_out": np.full((1, 1, 128), 2, dtype=np.float32),
            }

        async def close(self) -> None:
            return None

    async def fake_create_session(*_args: object, **_kwargs: object) -> FakeSession:
        return FakeSession()

    monkeypatch.setattr(
        "hugging_mac_sdk.models.silero.coreml.CoreMLProvider.create_session",
        fake_create_session,
    )
    config = SileroInstanceConfig(runtime="coreml", artifact_path=tmp_path / "model.mlmodelc")
    engine = CoreMlSileroEngine(config, object())  # type: ignore[arg-type]
    await engine.load(config.artifact_path)  # type: ignore[arg-type]

    result = await engine.probabilities(
        PreparedAudio(np.zeros(1024, dtype=np.float32), 16000, 0.064)
    )

    assert result == pytest.approx((0.3, 0.4))
    assert calls[0]["audio"].shape == (1, 1, 576)
    assert not calls[0]["h"].any()
    assert calls[1]["h"].all()
