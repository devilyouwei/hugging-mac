from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.models.kokoro import KOKORO_82M_MANIFEST, register_kokoro
from hugging_mac_sdk.models.kokoro.config import Kokoro82mInstanceConfig
from hugging_mac_sdk.models.kokoro.coreml import CoreMlKokoro82mEngine
from hugging_mac_sdk.models.kokoro.instance import Kokoro82mInstance
from hugging_mac_sdk.models.kokoro.resources import Kokoro82mResourceResolver
from hugging_mac_sdk.schemas.resources import HuggingFaceSource

PYTORCH_SOURCE = HuggingFaceSource(
    repo_id="hexgrad/Kokoro-82M",
    revision="f3ff3571791e39611d31c381e3a41a3af07b4987",
)
COREML_SOURCE = HuggingFaceSource(
    repo_id="FluidInference/kokoro-82m-coreml",
    revision="c94edcb4b671856795458645cd389c0a9184e8bb",
)


def test_kokoro_manifest_registers_downloadable_coreml_runtime() -> None:
    assert KOKORO_82M_MANIFEST.model_id == "hexgrad/kokoro"
    assert KOKORO_82M_MANIFEST.default_variant == "v1.0"
    assert KOKORO_82M_MANIFEST.default_runtime == "coreml"
    assert {runtime.name for runtime in KOKORO_82M_MANIFEST.runtimes} == {
        "coreml",
        "pytorch-mps",
    }
    models = ModelRegistry()
    definition = register_kokoro(models)
    assert definition.converter_ids == ()
    for runtime in ("coreml", "pytorch-mps"):
        instance = definition.create(runtime=runtime, variant="v1.0")
        assert isinstance(instance, Kokoro82mInstance)
        assert instance.supports(SpeechSynthesis)  # type: ignore[type-abstract]


async def test_kokoro_resolves_downloaded_coreml_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "coreml"
    for relative in (
        "kokoro_21_5s.mlmodelc/model.mil",
        "kokoro_21_5s.mlmodelc/weights/weight.bin",
        "vocab_index.json",
        "voices/af_heart.json",
    ):
        target = artifact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test")
    config = Kokoro82mInstanceConfig(artifact_path=artifact)
    resolver = Kokoro82mResourceResolver(PYTORCH_SOURCE, COREML_SOURCE, config)
    resolved = await resolver.resolve_coreml()
    assert resolved.path == artifact
    assert resolved.source == COREML_SOURCE
    status = resolver.status()
    coreml = next(item for item in status.artifacts if item.artifact_id == "coreml")
    assert coreml.provisioning == "download"
    assert status.conversion_targets == ()


def test_kokoro_coreml_splits_long_input_on_natural_boundaries() -> None:
    phonemes = f"{'a' * 80}, {'b' * 80}"

    chunks = CoreMlKokoro82mEngine._fit_input_chunks(phonemes)

    assert "".join(chunks) == phonemes.replace(" ", "")
    assert all(len(chunk) <= 122 for chunk in chunks)
    assert chunks[0].endswith(",")


def test_kokoro_coreml_preserves_low_energy_tail() -> None:
    audio = np.concatenate(
        (np.full(500, 0.2, dtype=np.float32), np.full(2_500, 0.01, dtype=np.float32))
    )

    prepared = CoreMlKokoro82mEngine._prepare_audio(audio, audio.size)

    assert prepared.size == audio.size
    assert np.allclose(prepared[500:2_500], 0.01)
    assert prepared[-1] == 0.0


def test_kokoro_coreml_trusts_fluid_reported_length() -> None:
    audio = np.zeros(24_000, dtype=np.float32)
    audio[12_000:15_600] = 0.05

    prepared = CoreMlKokoro82mEngine._prepare_audio(audio, 12_000)

    assert prepared.size == 12_000
    assert prepared[-1] == 0.0


def test_kokoro_coreml_retries_overflowing_audio_as_smaller_chunks() -> None:
    class FakeSession:
        device = "cpu-and-neural-engine"

        def __init__(self) -> None:
            self.token_counts: list[int] = []

        def run(self, inputs: dict[str, object]) -> dict[str, np.ndarray]:
            token_count = int(np.asarray(inputs["attention_mask"]).sum())
            self.token_counts.append(token_count)
            length = 170_000 if token_count > 50 else 2_400
            return {
                "audio": np.ones((1, 1, 175_800), dtype=np.float32),
                "audio_length_samples": np.asarray([length], dtype=np.int32),
                "pred_dur": np.ones((1, 124), dtype=np.float32),
            }

    config = Kokoro82mInstanceConfig(runtime="coreml")
    resolver = Kokoro82mResourceResolver(PYTORCH_SOURCE, COREML_SOURCE, config)
    engine = CoreMlKokoro82mEngine(config, resolver)
    session = FakeSession()
    engine._session = session  # type: ignore[assignment]
    engine._vocab = {"a": 3, "b": 4, " ": 5}

    waveforms, generated_tokens = engine._synthesize_phonemes(
        f"{'a' * 40} {'b' * 40}",
        embedding=np.zeros((1, 256), dtype=np.float32),
        cancelled=threading.Event(),
    )

    assert len(session.token_counts) == 3
    assert session.token_counts[0] > 50
    assert all(token_count <= 50 for token_count in session.token_counts[1:])
    assert [waveform.size for waveform in waveforms] == [2_400, 2_400]
    assert generated_tokens == sum(session.token_counts[1:])
