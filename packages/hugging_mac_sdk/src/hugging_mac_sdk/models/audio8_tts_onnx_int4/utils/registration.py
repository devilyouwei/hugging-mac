"""Transient reference-audio encoder for Audio8 voice profiles."""

from __future__ import annotations

import io
import json
from math import gcd
from pathlib import Path

import numpy as np

from hugging_mac_sdk.schemas.transcription import AudioInput


def encode_reference_audio(model_dir: Path, source: AudioInput) -> np.ndarray:
    import onnxruntime as ort  # type: ignore[import-untyped]
    import soundfile as sf  # type: ignore[import-untyped]
    from scipy.signal import resample_poly  # type: ignore[import-untyped]

    registration_dir = model_dir / "registration"
    manifest = json.loads(
        (registration_dir / "registration_manifest.json").read_text(encoding="utf-8")
    )
    audio_source: str | io.BytesIO
    if source.path is not None:
        audio_source = str(source.path)
    else:
        assert source.data is not None
        if len(source.data) > 50 * 1024 * 1024:
            raise ValueError("reference audio must not exceed 50 MiB")
        audio_source = io.BytesIO(source.data)
    try:
        audio, sample_rate = sf.read(audio_source, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise ValueError(f"unsupported or invalid reference audio: {error}") from error
    mono = audio.mean(axis=1)
    duration = mono.size / max(1, int(sample_rate))
    if duration < 0.5 or duration > 30.0:
        raise ValueError("reference audio duration must be between 0.5 and 30 seconds")
    if not np.isfinite(mono).all():
        raise ValueError("reference audio contains non-finite samples")
    target_rate = int(manifest["sample_rate"])
    if int(sample_rate) != target_rate:
        factor = gcd(int(sample_rate), target_rate)
        mono = resample_poly(mono, target_rate // factor, int(sample_rate) // factor).astype(
            np.float32
        )
    padding = (-mono.size) % int(manifest.get("frame_length", 2048))
    if padding:
        mono = np.pad(mono, (0, padding))

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(registration_dir / "codec_encoder_fp16.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    dtype = np.float16 if input_info.type == "tensor(float16)" else np.float32
    values = np.ascontiguousarray(mono.reshape(1, 1, -1), dtype=dtype)
    codes = np.asarray(session.run(None, {input_info.name: values})[0], dtype=np.int64)
    del session
    if codes.ndim == 3:
        codes = codes[0]
    expected_codebooks = int(manifest["num_codebooks"])
    if codes.ndim != 2 or codes.shape[0] != expected_codebooks or codes.shape[1] == 0:
        raise RuntimeError(f"reference encoder returned invalid codes: {codes.shape}")
    return codes
