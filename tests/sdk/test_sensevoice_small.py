from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hugging_mac_sdk.capabilities import SpeechTranscription, SpeechUnderstanding
from hugging_mac_sdk.core.instance import ModelState
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.sensevoice import (
    SENSEVOICE_SMALL_DEFINITION,
    SENSEVOICE_SMALL_MANIFEST,
    register_sensevoice,
)
from hugging_mac_sdk.models.sensevoice.config import (
    SenseVoiceSmallInstanceConfig,
)
from hugging_mac_sdk.models.sensevoice.coreml import CoreMlSenseVoiceSmallEngine
from hugging_mac_sdk.models.sensevoice.instance import SenseVoiceSmallInstance
from hugging_mac_sdk.models.sensevoice.resources import (
    SenseVoiceSmallResourceProvider,
)
from hugging_mac_sdk.models.sensevoice.utils.postprocess import (
    parse_rich_transcript,
)
from hugging_mac_sdk.models.sensevoice.utils.types import (
    PreparedAudio,
    SenseVoiceEngineOutput,
    SenseVoiceInferenceOptions,
)
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.speech_understanding import SpeechUnderstandingRequest
from hugging_mac_sdk.schemas.transcription import (
    AudioInput,
    TranscriptionRequest,
)


class FakeSenseVoiceEngine:
    runtime_name = "pytorch-mps"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.loaded = False
        self.closed = False
        self.options: list[SenseVoiceInferenceOptions] = []

    @property
    def device(self) -> str:
        return "mps"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact
        self.loaded = True

    async def infer(
        self,
        prepared: PreparedAudio,
        options: SenseVoiceInferenceOptions,
    ) -> SenseVoiceEngineOutput:
        assert prepared.sample_rate == 16000
        self.options.append(options)
        return SenseVoiceEngineOutput(
            text="你好 Hugging Mac 😊",
            raw_text=("<|zh|><|HAPPY|><|Speech|><|withitn|>你好 Hugging Mac"),
            languages=("zh",),
            emotion="happy",
            events=("speech",),
            token_count=8,
        )

    async def close(self) -> None:
        self.closed = True


class FakeCoreMlSession:
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.closed = False

    @property
    def device(self) -> str:
        return "cpu-and-neural-engine"

    def run(self, inputs: object) -> object:
        return inputs

    async def close(self) -> None:
        self.closed = True


def test_sensevoice_manifest_pins_minimal_snapshot_and_license() -> None:
    assert SENSEVOICE_SMALL_MANIFEST.default_variant == "small"
    assert SENSEVOICE_SMALL_MANIFEST.capabilities == {
        "speech-transcription",
        "speech-understanding",
    }
    assert SENSEVOICE_SMALL_MANIFEST.license == "FunASR-Model-License-1.1"
    assert {runtime.name for runtime in SENSEVOICE_SMALL_MANIFEST.runtimes} == {
        "pytorch-mps",
        "coreml",
    }
    source = SENSEVOICE_SMALL_MANIFEST.get_variant().resources[0]
    assert source.allow_patterns == (  # type: ignore[union-attr]
        "model.pt",
        "config.yaml",
        "configuration.json",
        "am.mvn",
    )
    shared = SENSEVOICE_SMALL_DEFINITION.shared_artifacts[0]
    assert shared.artifact_id == "tokenizer"
    assert shared.source.allow_patterns == (  # type: ignore[union-attr]
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
    )


def test_registered_factory_exposes_both_speech_capabilities() -> None:
    registry = ModelRegistry()
    definition = register_sensevoice(registry)

    instance = definition.create(
        runtime="pytorch-mps",
        variant="small",
        options={"device": "cpu"},
    )

    assert isinstance(instance, SenseVoiceSmallInstance)
    assert instance.supports(SpeechTranscription)  # type: ignore[type-abstract]
    assert instance.supports(SpeechUnderstanding)  # type: ignore[type-abstract]
    assert instance.info().variant == "small"
    assert instance.info().runtime == "pytorch-mps"
    assert instance.info().device == "cpu"


async def test_instance_composes_plain_and_rich_speech_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    engine = FakeSenseVoiceEngine(artifact)
    prepared = PreparedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.sensevoice.instance.prepare_audio",
        lambda *_args, **_kwargs: prepared,
    )
    instance = SenseVoiceSmallInstance(SenseVoiceSmallInstanceConfig(), engine)

    await instance.load()
    transcriber = instance.require(SpeechTranscription)  # type: ignore[type-abstract]
    transcription = await transcriber.transcribe(
        TranscriptionRequest(audio=AudioInput(data=b"encoded audio"))
    )
    understanding = instance.require(  # type: ignore[type-abstract]
        SpeechUnderstanding
    )
    rich = await understanding.understand_speech(
        SpeechUnderstandingRequest(
            audio=AudioInput(data=b"encoded audio"),
            language="zh",
            use_itn=False,
            ban_unknown_emotion=True,
        )
    )

    assert instance.state is ModelState.READY
    assert engine.loaded
    assert transcription.text == "你好 Hugging Mac 😊"
    assert transcription.generated_tokens == 8
    assert rich.languages == ("zh",)
    assert rich.emotion == "happy"
    assert rich.events == ("speech",)
    assert rich.token_count == 8
    assert rich.timings.inference_ms is not None
    assert engine.options == [
        SenseVoiceInferenceOptions(language="auto", use_itn=True),
        SenseVoiceInferenceOptions(
            language="zh",
            use_itn=False,
            ban_unknown_emotion=True,
        ),
    ]

    await instance.unload()
    assert engine.closed
    assert instance.state is ModelState.UNLOADED


def test_rich_postprocess_returns_text_and_structured_annotations() -> None:
    result = parse_rich_transcript("<|en|><|HAPPY|><|Laughter|><|withitn|>Hello!")

    assert result.text == "😀Hello!😊"
    assert result.raw_text.endswith("Hello!")
    assert result.languages == ("en",)
    assert result.emotion == "happy"
    assert result.events == ("laughter",)


async def test_resource_provider_routes_coreml_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SENSEVOICE_SMALL_MANIFEST.get_variant().resources[0]
    provider = SenseVoiceSmallResourceProvider(source)  # type: ignore[arg-type]
    converted: list[bool] = []

    async def fake_convert(self: object, *, overwrite: bool = False) -> None:
        converted.append(overwrite)

    monkeypatch.setattr(
        "hugging_mac_sdk.models.sensevoice.resources."
        "SenseVoiceSmallResourceResolver.convert_coreml",
        fake_convert,
    )
    monkeypatch.setattr(
        "hugging_mac_sdk.models.sensevoice.resources.SenseVoiceSmallResourceResolver.status",
        lambda self: "status",
    )

    result = await provider.convert("small", ArtifactFormat.COREML, overwrite=True)

    assert result == "status"
    assert converted == [True]


async def test_coreml_engine_lazily_reuses_only_one_bucket_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeCoreMlSession] = []

    async def fake_create_session(
        self: object,
        artifact: Path,
        *,
        device: str | None,
        options: dict[str, object],
        executor: object | None = None,
    ) -> FakeCoreMlSession:
        del self, artifact, device, executor
        session = FakeCoreMlSession(str(options["function_name"]))
        created.append(session)
        return session

    monkeypatch.setattr(
        "hugging_mac_sdk.models.sensevoice.coreml.CoreMLProvider.create_session",
        fake_create_session,
    )
    config = SenseVoiceSmallInstanceConfig(runtime="coreml")
    engine = CoreMlSenseVoiceSmallEngine(config, object())  # type: ignore[arg-type]
    engine._artifact = tmp_path

    first = await engine._resolve_session(100)
    repeated = await engine._resolve_session(100)
    second_bucket = await engine._resolve_session(250)

    assert first is repeated
    assert first is not second_bucket
    assert created[0].closed
    assert not created[1].closed
    assert [session.function_name for session in created] == ["encoder_100", "encoder_250"]
