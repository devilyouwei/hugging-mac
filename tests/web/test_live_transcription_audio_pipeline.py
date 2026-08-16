from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from hugging_mac_sdk.schemas.voice_activity import SpeechSegment
from hugging_mac_web.live_transcription.audio_pipeline import prepare_detected_speech


def test_prepare_detected_speech_resamples_merges_and_removes_silence() -> None:
    sample_rate = 48000
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[12000:19200] = 0.2
    samples[21600:28800] = -0.2
    encoded = io.BytesIO()
    sf.write(encoded, samples, sample_rate, format="WAV", subtype="PCM_16")

    prepared = prepare_detected_speech(
        encoded.getvalue(),
        (
            SpeechSegment(
                start_sample=4000,
                end_sample=6400,
                start_seconds=0.25,
                end_seconds=0.4,
            ),
            SpeechSegment(
                start_sample=7200,
                end_sample=9600,
                start_seconds=0.45,
                end_seconds=0.6,
            ),
        ),
        vad_sample_rate=16000,
    )

    decoded, decoded_rate = sf.read(io.BytesIO(prepared.wav), dtype="float32")
    assert decoded_rate == 16000
    assert prepared.source_duration_seconds == 1.0
    assert prepared.segment_count == 1
    assert 0.55 < prepared.speech_duration_seconds < 0.7
    assert len(decoded) == round(prepared.speech_duration_seconds * decoded_rate)
    assert abs(float(np.mean(decoded))) < 1e-4
