from __future__ import annotations

from array import array

from hugging_mac_web.live_transcription.session import (
    FRAME_SAMPLES,
    AudioFrame,
    EnergyDetector,
    FrameResampler,
    _audio_message,
    _encode_wav,
)


def test_resampler_emits_immutable_aligned_frames_with_monotonic_offsets() -> None:
    resampler = FrameResampler(16_000)
    samples = array("h", range(FRAME_SAMPLES * 2 + 1)).tobytes()

    frames = resampler.push(samples)

    assert len(frames) == 2
    assert [frame.start_sample for frame in frames] == [0, FRAME_SAMPLES]
    assert all(isinstance(frame.pcm16, bytes) for frame in frames)
    assert all(len(frame.pcm16) == FRAME_SAMPLES * 2 for frame in frames)


async def test_energy_detector_uses_speech_hysteresis_and_640ms_endpoint() -> None:
    detector = EnergyDetector(0.65)
    speech = tuple([0.2] * FRAME_SAMPLES)
    silence = tuple([0.0] * FRAME_SAMPLES)

    for _ in range(2):
        assert not (await detector.push(speech)).speech_started
    assert (await detector.push(speech)).speech_started
    for _ in range(19):
        assert not (await detector.push(silence)).speech_ended
    assert (await detector.push(silence)).speech_ended


def test_wav_and_binary_result_reuse_the_same_encoded_audio() -> None:
    frame = AudioFrame(0, b"\x01\x00" * FRAME_SAMPLES)

    wav = _encode_wav((frame,))
    message = _audio_message(42, wav)

    assert wav.startswith(b"RIFF")
    assert message[:8] == (42).to_bytes(8, "big")
    assert message[8:] == wav
    assert len(wav) == 44 + FRAME_SAMPLES * 2
