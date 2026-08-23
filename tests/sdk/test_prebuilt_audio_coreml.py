from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hugging_mac_sdk import ModelSdk
from hugging_mac_sdk.capabilities import SpeechEnhancement, SpeechTranscription
from hugging_mac_sdk.models.deepfilternet3 import DEEPFILTERNET3_DEFINITION, DEEPFILTERNET3_MANIFEST
from hugging_mac_sdk.models.deepfilternet3.audio import PreparedEnhancementAudio
from hugging_mac_sdk.models.deepfilternet3.config import DeepFilterNet3InstanceConfig
from hugging_mac_sdk.models.deepfilternet3.dsp import (
    DF_BINS,
    DF_ORDER,
    ERB_BANDS,
    FFT_SIZE,
    FREQUENCY_BINS,
    DeepFilterNet3Dsp,
    EnhancementOutput,
)
from hugging_mac_sdk.models.deepfilternet3.instance import DeepFilterNet3Instance
from hugging_mac_sdk.models.qwen3_asr import (
    QWEN3_ASR_DEFINITION,
    QWEN3_ASR_MANIFEST,
    register_qwen3_asr,
)
from hugging_mac_sdk.models.qwen3_asr.config import Qwen3AsrCoreMlInstanceConfig
from hugging_mac_sdk.models.qwen3_asr.instance import Qwen3AsrCoreMlInstance
from hugging_mac_sdk.models.qwen3_asr.utils.types import AsrEngineOutput, PreparedAudio
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.resources import HuggingFaceSource, ResolvedResource
from hugging_mac_sdk.schemas.speech_enhancement import SpeechEnhancementRequest
from hugging_mac_sdk.schemas.transcription import AudioInput, TranscriptionRequest


class FakeQwenAsrEngine:
    runtime_name = "coreml"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    @property
    def device(self) -> str:
        return "cpu-and-neural-engine"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact

    async def infer(
        self, prepared: PreparedAudio, request: TranscriptionRequest
    ) -> AsrEngineOutput:
        assert prepared.sample_rate == 16000
        return AsrEngineOutput(" Qwen transcription ", 3, 19)

    async def close(self) -> None:
        return None


class FakeDeepFilterNet3Engine:
    runtime_name = "coreml"

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    @property
    def device(self) -> str:
        return "all"

    async def resolve(self) -> Path:
        return self.artifact

    async def load(self, artifact: Path) -> None:
        assert artifact == self.artifact

    async def enhance(self, prepared: PreparedEnhancementAudio) -> EnhancementOutput:
        return EnhancementOutput(prepared.samples.copy(), 0.1)

    async def close(self) -> None:
        return None


def test_deepfilternet3_declares_prebuilt_coreml_artifact_source() -> None:
    definition = DEEPFILTERNET3_DEFINITION

    assert definition.manifest.default_runtime == "coreml"
    assert definition.manifest.revision == "main"
    assert definition.artifacts[0].artifact_id == "coreml-int8"
    assert definition.artifacts[0].source.repo_id == "aufklarer/DeepFilterNet3-CoreML"


def test_deepfilternet3_dsp_identity_path_preserves_signal_and_length(tmp_path: Path) -> None:
    indices = np.arange(FFT_SIZE, dtype=np.float32)
    window = np.sin(np.pi / 2 * np.square(np.sin(np.pi * (indices + 0.5) / FFT_SIZE))).astype(
        np.float32
    )
    auxiliary = tmp_path / "auxiliary.npz"
    np.savez(
        auxiliary,
        erb_fb=np.full((FREQUENCY_BINS, ERB_BANDS), 1 / FREQUENCY_BINS, np.float32),
        erb_inv_fb=np.ones((ERB_BANDS, FREQUENCY_BINS), np.float32),
        window=window,
        mean_norm_state=np.full(ERB_BANDS, -60, np.float32),
        unit_norm_state=np.full((1, DF_BINS), 0.001, np.float32),
    )
    dsp = DeepFilterNet3Dsp(auxiliary)
    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(12345) / 48000)).astype(np.float32)

    def identity_predict(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        frames = inputs["feat_erb"].shape[2]
        coefficients = np.zeros((1, DF_ORDER, frames, DF_BINS, 2), dtype=np.float32)
        coefficients[:, 2, :, :, 0] = 1
        return {
            "erb_mask": np.full((1, 1, frames, ERB_BANDS), 1 / ERB_BANDS, np.float32),
            "df_coefs": coefficients,
        }

    result = dsp.enhance(samples, identity_predict)

    assert result.samples.shape == samples.shape
    assert np.max(np.abs(result.samples - samples)) < 2e-6


async def test_deepfilternet3_exposes_speech_enhancement(tmp_path: Path) -> None:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    instance = DeepFilterNet3Instance(
        DeepFilterNet3InstanceConfig(artifact_path=tmp_path),
        FakeDeepFilterNet3Engine(tmp_path),
        DEEPFILTERNET3_MANIFEST,
    )
    assert instance.supports(SpeechEnhancement)  # type: ignore[type-abstract]
    await instance.load()

    response = await instance.require(SpeechEnhancement).enhance_speech(  # type: ignore[type-abstract]
        SpeechEnhancementRequest(audio=AudioInput(data=stream.getvalue()))
    )

    enhanced, sample_rate = sf.read(io.BytesIO(response.audio), dtype="float32")
    assert sample_rate == 16000
    assert enhanced.shape == (1600,)
    assert response.runtime == "coreml"


def test_qwen3_asr_declares_all_prebuilt_coreml_graphs() -> None:
    definition = QWEN3_ASR_DEFINITION
    source = definition.artifacts[0].source

    assert definition.manifest.default_runtime == "coreml"
    assert definition.manifest.revision == "main"
    assert len(definition.artifacts) == 2
    assert isinstance(source, HuggingFaceSource)
    assert source.repo_id == "aufklarer/Qwen3-ASR-CoreML"
    assert source.allow_patterns == (
        "encoder.mlmodelc/**",
        "embedding.mlmodelc/**",
        "decoder_part1.mlmodelc/**",
        "decoder_part2.mlmodelc/**",
        "config.json",
    )
    tokenizer = definition.artifacts[1]
    assert tokenizer.shared
    assert isinstance(tokenizer.source, HuggingFaceSource)
    assert tokenizer.source.repo_id == "Qwen/Qwen3-ASR-0.6B"
    assert tokenizer.source.allow_patterns == (
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    )


async def test_qwen3_asr_reports_one_bundle_only_when_graphs_and_tokenizer_exist(
    tmp_path: Path,
) -> None:
    provider = QWEN3_ASR_DEFINITION.resource_provider
    assert provider is not None
    model = next(
        artifact
        for artifact in QWEN3_ASR_DEFINITION.artifacts
        if artifact.artifact_id == "coreml-int8"
    ).resolve(tmp_path)
    for graph in ("encoder", "embedding", "decoder_part1", "decoder_part2"):
        path = model / f"{graph}.mlmodelc/model.mil"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"mil")
    (model / "config.json").write_text("{}")

    missing_tokenizer = await provider.status("0.6b", {"model_home": tmp_path})
    assert len(missing_tokenizer.artifacts) == 2
    assert not missing_tokenizer.runtimes[0].available

    tokenizer = next(
        artifact
        for artifact in QWEN3_ASR_DEFINITION.artifacts
        if artifact.artifact_id == "tokenizer"
    ).resolve(tmp_path)
    tokenizer.mkdir(parents=True)
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        (tokenizer / name).write_text("{}")
    ready = await provider.status("0.6b", {"model_home": tmp_path})

    assert len(ready.artifacts) == 2
    assert ready.artifacts[0].artifact_id == "coreml-int8"
    assert ready.artifacts[1].artifact_id == "tokenizer"
    assert ready.runtimes[0].available
    assert ready.runtimes[0].artifact_ids == ("coreml-int8", "tokenizer")

    deleted = await provider.delete("0.6b", {"model_home": tmp_path}, runtime="coreml")
    assert not deleted.artifacts[0].available
    assert deleted.artifacts[1].available
    assert not deleted.runtimes[0].available
    assert tokenizer.is_dir()


async def test_qwen3_asr_downloads_shared_tokenizer_first_and_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[str] = []

    async def fake_download(
        _: ResourceDownloader,
        source: HuggingFaceSource,
        destination: Path,
        **__: object,
    ) -> ResolvedResource:
        downloads.append(source.repo_id)
        destination.mkdir(parents=True)
        if source.repo_id == "Qwen/Qwen3-ASR-0.6B":
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
                (destination / name).write_text("{}")
        else:
            for graph in ("encoder", "embedding", "decoder_part1", "decoder_part2"):
                path = destination / f"{graph}.mlmodelc/model.mil"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"mil")
            (destination / "config.json").write_text("{}")
        return ResolvedResource(
            path=destination,
            source=source,
            size_bytes=directory_size(destination),
        )

    monkeypatch.setattr(ResourceDownloader, "download", fake_download)
    sdk = ModelSdk()
    register_qwen3_asr(sdk.registry)
    options = {"model_home": tmp_path}

    first = await sdk.resources.download_source("qwen/qwen3-asr", options=options)
    second = await sdk.resources.download_source("qwen/qwen3-asr", options=options)

    assert downloads == ["Qwen/Qwen3-ASR-0.6B", "aufklarer/Qwen3-ASR-CoreML"]
    assert first.runtimes[0].available
    assert second.runtimes[0].available


async def test_qwen3_asr_exposes_speech_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = PreparedAudio(np.zeros(16000, dtype=np.float32), 16000, 1.0)
    monkeypatch.setattr(
        "hugging_mac_sdk.models.qwen3_asr.instance.prepare_audio",
        lambda *_args, **_kwargs: prepared,
    )
    instance = Qwen3AsrCoreMlInstance(
        Qwen3AsrCoreMlInstanceConfig(), FakeQwenAsrEngine(tmp_path), QWEN3_ASR_MANIFEST
    )
    assert instance.supports(SpeechTranscription)  # type: ignore[type-abstract]
    await instance.load()
    response = await instance.require(SpeechTranscription).transcribe(  # type: ignore[type-abstract]
        TranscriptionRequest(audio=AudioInput(data=b"wav"))
    )
    assert response.text == "Qwen transcription"
    assert response.generated_tokens == 3
    assert response.prompt_tokens == 19
