"""SenseVoice Kaldi-fbank, low-frame-rate stacking, and CMVN frontend."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi  # type: ignore[import-untyped]

from .types import PreparedAudio


def load_cmvn(path: Path) -> torch.Tensor:
    means: list[float] = []
    scales: list[float] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if not fields or index + 1 >= len(lines):
            continue
        values = lines[index + 1].split()
        if fields[0] == "<AddShift>" and values[:1] == ["<LearnRateCoef>"]:
            means = [float(value) for value in values[3:-1]]
        elif fields[0] == "<Rescale>" and values[:1] == ["<LearnRateCoef>"]:
            scales = [float(value) for value in values[3:-1]]
    if not means or len(means) != len(scales):
        raise ValueError(f"SenseVoice CMVN file is invalid: {path}")
    return torch.tensor((means, scales), dtype=torch.float32)


def apply_lfr(inputs: torch.Tensor, *, width: int = 7, stride: int = 6) -> torch.Tensor:
    """Stack seven adjacent fbank frames every six frames."""

    frame_count = int(inputs.shape[0])
    output_count = (frame_count + stride - 1) // stride
    left_padding = inputs[0].repeat((width - 1) // 2, 1)
    padded = torch.vstack((left_padding, inputs))
    padded_count = int(padded.shape[0])
    feature_size = int(padded.shape[-1])
    last_index = (padded_count - width) // stride + 1
    padding = width - (padded_count - last_index * stride)
    if padding > 0:
        padding = int(
            (2 * width - 2 * padded_count + (output_count - 1 + last_index) * stride)
            / 2
            * (output_count - last_index)
        )
        padded = torch.vstack([padded] + [padded[-1:]] * padding)
    return padded.as_strided(
        (output_count, width * feature_size),
        (stride * feature_size, 1),
    ).clone()


def extract_features(
    prepared: PreparedAudio,
    *,
    cmvn_path: Path,
    dither: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one padded [batch, frames, 560] feature tensor and its length."""

    waveform = torch.from_numpy(prepared.samples).float() * (1 << 15)
    waveform = waveform.unsqueeze(0)
    frame_length = min(
        25.0,
        max(float(waveform.shape[-1]) / prepared.sample_rate * 1000.0, 1.0),
    )
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=frame_length,
        frame_shift=10.0,
        dither=dither,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=prepared.sample_rate,
        snip_edges=True,
    )
    if not int(features.shape[0]):
        raise ValueError("Audio is too short to extract SenseVoice features")
    features = apply_lfr(features)
    cmvn = load_cmvn(cmvn_path)
    if int(cmvn.shape[1]) < int(features.shape[1]):
        raise ValueError("SenseVoice CMVN dimension does not match fbank features")
    features = (features + cmvn[0, : features.shape[1]]) * cmvn[1, : features.shape[1]]
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
    return features.unsqueeze(0).float(), lengths
