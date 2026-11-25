"""Portable streaming host for the Zipformer Core ML graph bundle."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import wave
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
Predict = Callable[[str, tuple[Array, ...]], tuple[Array, ...]]
RATE = 16000
FRAME_SECONDS = 0.04
INITIAL_CONTEXT = (-1, 0)


@dataclass(frozen=True)
class Keyword:
    label: str
    boost: float
    threshold: float


@dataclass(frozen=True)
class Hit:
    label: str
    frames: tuple[int, ...]


@dataclass(frozen=True)
class Hypothesis:
    tokens: tuple[int, ...] = INITIAL_CONTEXT
    prefix: tuple[int, ...] = ()
    frames: tuple[int, ...] = ()
    probabilities: tuple[float, ...] = ()
    blanks: int = 0
    score: float = 0.0


def read_keywords(path: Path, tokens_path: Path) -> dict[tuple[int, ...], Keyword]:
    symbols = {
        symbol: int(index) for symbol, index in map(str.split, tokens_path.read_text().splitlines())
    }
    keywords = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        words: list[str] = []
        label, boost, threshold = "", 1.0, 0.25
        for field in line.split():
            marker, value = field[0], field[1:]
            if field in symbols:
                words.append(field)
            elif marker == "@":
                label = value
            elif marker == ":":
                boost = float(value)
            elif marker == "#":
                threshold = float(value)
            else:
                raise ValueError(f"Unknown token: {field}")
        ids = tuple(symbols[word] for word in words)
        if not ids or any(token in (0, 2) for token in ids) or ids in keywords:
            raise ValueError(f"Expected unique nonblank keyword tokens: {line}")
        if not math.isfinite(boost) or boost < 0 or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid keyword boost or threshold: {line}")
        keywords[ids] = Keyword(label or " ".join(words), boost, threshold)
    if not keywords:
        raise ValueError("At least one keyword is required")
    return keywords


class Beam:
    def __init__(self, predict: Predict, keywords: dict[tuple[int, ...], Keyword]) -> None:
        self.predict = predict
        self.keywords = keywords
        self.hypotheses = [Hypothesis()]
        self.steps = 0
        self.history = max(map(len, keywords)) + len(INITIAL_CONTEXT)
        edge_scores: dict[tuple[int, ...], float] = {}
        for ids, keyword in keywords.items():
            for size in range(1, len(ids) + 1):
                prefix = ids[:size]
                edge_scores[prefix] = max(edge_scores.get(prefix, 0.0), keyword.boost)
        self.scores: dict[tuple[int, ...], float] = {(): 0.0}
        for prefix in sorted(edge_scores, key=len):
            self.scores[prefix] = self.scores[prefix[:-1]] + edge_scores[prefix]
        self.matches = {
            prefix: tuple(
                prefix[start:] for start in range(len(prefix)) if prefix[start:] in keywords
            )
            for prefix in self.scores
        }
        self.decode = lru_cache(maxsize=128)(self._decode)

    def _decode(self, ids: tuple[int, ...]) -> Array:
        return self.predict("decoder", (np.asarray([ids], dtype=np.int32),))[0]

    def advance(
        self, parent: Hypothesis, token: int, score: float, probability: float
    ) -> Hypothesis:
        if token in (0, 2):
            return Hypothesis(
                parent.tokens,
                parent.prefix,
                parent.frames,
                parent.probabilities,
                parent.blanks + 1,
                score,
            )
        prefix = (*parent.prefix, token)
        while prefix not in self.scores:
            prefix = prefix[1:]
        score += self.scores[prefix] - self.scores[parent.prefix]
        score += sum(self.scores[match] for match in self.matches[prefix])
        if not prefix:
            return Hypothesis(score=score)
        return Hypothesis(
            (*parent.tokens, token)[-self.history :],
            prefix,
            (*parent.frames, self.steps)[-self.history :],
            (*parent.probabilities, probability)[-self.history :],
            0,
            score,
        )

    def step(self, frame: Array, trailing_blanks: int) -> Hit | None:
        logits = np.stack(
            [
                self.predict("joiner", (frame, self.decode(parent.tokens[-2:])))[0][0]
                for parent in self.hypotheses
            ]
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        scores = probabilities + np.asarray([parent.score for parent in self.hypotheses])[:, None]
        merged: dict[tuple[int, ...], Hypothesis] = {}
        for flat in np.argsort(-scores, axis=None, kind="stable")[:4]:
            row, token = divmod(int(flat), logits.shape[1])
            candidate = self.advance(
                self.hypotheses[row],
                token,
                float(scores[row, token]),
                math.exp(float(probabilities[row, token])),
            )
            existing = merged.get(candidate.tokens)
            if existing is None:
                merged[candidate.tokens] = candidate
            else:
                merged[candidate.tokens] = Hypothesis(
                    existing.tokens,
                    existing.prefix,
                    existing.frames,
                    existing.probabilities,
                    existing.blanks,
                    float(np.logaddexp(existing.score, candidate.score)),
                )
        self.hypotheses = list(merged.values())
        best = max(self.hypotheses, key=lambda h: h.score)
        self.steps += 1
        if best.blanks <= trailing_blanks:
            return None
        for match in self.matches[best.prefix]:
            if (
                sum(best.probabilities[-len(match) :]) / len(match)
                >= self.keywords[match].threshold
            ):
                self.hypotheses = [Hypothesis()]
                return Hit(self.keywords[match].label, best.frames[-len(match) :])
        return None


class CoreML:
    def __init__(self, bundle: Path, contract: dict[str, Any], compute_unit: str) -> None:
        ct = importlib.import_module("coremltools")
        units = getattr(ct.ComputeUnit, compute_unit)
        self.contract = contract
        self.models = {
            part: ct.models.MLModel(str(bundle / f"{part}.mlpackage"), compute_units=units)
            for part in contract
        }

    def __call__(self, part: str, values: tuple[Array, ...]) -> tuple[Array, ...]:
        metadata = self.contract[part]
        features = {
            info["name"]: np.asarray(value, dtype=info["dtype"])
            for info, value in zip(metadata["inputs"], values, strict=True)
        }
        prediction = self.models[part].predict(features)
        return tuple(prediction[name] for name in metadata["outputs"])


class Stream:
    def __init__(
        self,
        predict: Predict,
        manifest: dict[str, Any],
        keywords: dict[tuple[int, ...], Keyword],
        trailing_blanks: int = 0,
    ) -> None:
        if trailing_blanks < 0:
            raise ValueError("Trailing blanks must be nonnegative")
        knf = importlib.import_module("kaldi_native_fbank")
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = RATE
        options.frame_opts.snip_edges = False
        options.frame_opts.dither = 0.0
        options.mel_opts.num_bins = 80
        options.mel_opts.low_freq = 20.0
        options.mel_opts.high_freq = -400.0
        self.frontend = knf.OnlineFbank(options)
        self.cache = tuple(
            np.zeros(entry["shape"], dtype=entry["dtype"])
            for entry in manifest["graphs"]["encoder"]["inputs"][1:]
        )
        self.predict = predict
        self.beam = Beam(predict, keywords)
        self.window = manifest["features_with_lookahead"]
        self.hop = manifest["features_per_chunk"]
        self.frame = 0
        self.trailing_blanks = trailing_blanks

    def accept(self, samples: Array) -> list[Hit]:
        self.frontend.accept_waveform(RATE, samples.tolist())
        hits = []
        while self.frame + self.window <= self.frontend.num_frames_ready:
            frames = np.asarray(
                [
                    self.frontend.get_frame(index)
                    for index in range(self.frame, self.frame + self.window)
                ],
                dtype=np.float32,
            )[None]
            output, *cache = self.predict("encoder", (frames, *self.cache))
            self.cache = tuple(cache)
            self.frontend.pop(self.hop)
            self.frame += self.hop
            for frame in output[0]:
                hit = self.beam.step(frame[None], self.trailing_blanks)
                if hit is not None:
                    hits.append(hit)
        return hits


def read_pcm(wav: Path) -> Array:
    with wave.open(str(wav), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
            raise ValueError("Expected a 16 kHz mono PCM16 WAV")
        return (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )


def spot(
    predict: Predict,
    manifest: dict[str, Any],
    keywords: dict[tuple[int, ...], Keyword],
    samples: Array,
    trailing_blanks: int = 0,
) -> list[Hit]:
    stream = Stream(predict, manifest, keywords, trailing_blanks)
    samples = np.pad(samples, (0, RATE // 2))
    return [
        hit
        for start in range(0, samples.size, 512)
        for hit in stream.accept(samples[start : start + 512])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--keywords", type=Path)
    parser.add_argument("--compute-unit", choices=("CPU_ONLY", "ALL"), default="CPU_ONLY")
    parser.add_argument("--trailing-blanks", type=int, default=0)
    args = parser.parse_args()
    bundle = args.bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    keywords = read_keywords(args.keywords or bundle / "keywords.txt", bundle / "tokens.txt")
    native = CoreML(bundle, manifest["graphs"], args.compute_unit)
    for hit in spot(native, manifest, keywords, read_pcm(args.wav), args.trailing_blanks):
        print(
            json.dumps(
                {
                    "keyword": hit.label,
                    "token_seconds": [round(frame * FRAME_SECONDS, 2) for frame in hit.frames],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
