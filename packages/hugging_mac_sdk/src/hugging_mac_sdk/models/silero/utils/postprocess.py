"""Turn frame probabilities into padded speech intervals."""

from __future__ import annotations

from .types import SampleInterval, SileroInferenceOptions


def probabilities_to_segments(
    probabilities: tuple[float, ...],
    *,
    total_samples: int,
    sample_rate: int,
    chunk_samples: int,
    options: SileroInferenceOptions,
) -> tuple[SampleInterval, ...]:
    min_speech = round(sample_rate * options.min_speech_ms / 1000)
    min_silence = round(sample_rate * options.min_silence_ms / 1000)
    pad = round(sample_rate * options.speech_pad_ms / 1000)
    max_speech = (
        round(sample_rate * options.max_speech_seconds)
        if options.max_speech_seconds is not None
        else None
    )
    negative_threshold = max(options.threshold - 0.15, 0.01)
    raw: list[SampleInterval] = []
    start: int | None = None
    silence_start: int | None = None

    for index, probability in enumerate(probabilities):
        position = index * chunk_samples
        if probability >= options.threshold:
            silence_start = None
            if start is None:
                start = position
        elif start is not None and probability < negative_threshold:
            silence_start = silence_start if silence_start is not None else position
            if position - silence_start >= min_silence:
                if silence_start - start >= min_speech:
                    raw.append(SampleInterval(start, silence_start))
                start = None
                silence_start = None
        if start is not None and max_speech is not None and position - start >= max_speech:
            raw.append(SampleInterval(start, min(position, total_samples)))
            start = None
            silence_start = None
    if start is not None and total_samples - start >= min_speech:
        raw.append(SampleInterval(start, total_samples))

    padded: list[SampleInterval] = []
    for segment in raw:
        current = SampleInterval(max(0, segment.start - pad), min(total_samples, segment.end + pad))
        if padded and current.start <= padded[-1].end:
            padded[-1] = SampleInterval(padded[-1].start, max(padded[-1].end, current.end))
        elif current.end > current.start:
            padded.append(current)
    return tuple(padded)
