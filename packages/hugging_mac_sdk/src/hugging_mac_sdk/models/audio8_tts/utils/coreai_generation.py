"""Framework-free prompt construction and sampling for native Audio8 graphs."""

from __future__ import annotations

import importlib
import re
from typing import Any


def prompt_segments(
    tokenizer: Any, text: str, reference_text: str | None
) -> tuple[list[int], list[int]]:
    target = " ".join(text.strip().split())
    if not target:
        raise ValueError("text must not be empty")

    def encode(parts: list[str]) -> list[int]:
        return [
            token
            for part in parts
            for token in tokenizer.encode(part, add_special_tokens=False).ids
        ]

    tail = [
        "<|im_end|>\n",
        "<|im_start|>user\n",
        target,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    if reference_text is None:
        return encode(["<|im_start|>system\n", "convert the provided text to speech", *tail]), []
    reference = " ".join(reference_text.strip().split())
    if not re.search(r"<\|speaker:\d+\|>", reference):
        reference = f"<|speaker:0|>{reference}"
    return encode(
        [
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            reference,
            "\n\nSpeech:\n",
        ]
    ), encode(tail)


def sample_token(
    logits: Any, *, top_k: int, top_p: float, temperature: float, do_sample: bool, rng: Any
) -> int:
    np = importlib.import_module("numpy")
    scores = np.asarray(logits, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    probabilities = np.exp(ranked - ranked[0])
    probabilities /= probabilities.sum()
    remove = np.cumsum(probabilities) > top_p
    if top_k > 0:
        remove |= np.arange(len(ranked)) >= top_k
    remove[0] = False
    filtered = scores.copy()
    filtered[order[remove]] = -np.inf
    if not do_sample:
        return int(filtered.argmax())
    filtered /= max(temperature, 1e-5)
    probabilities = np.exp(filtered - filtered.max())
    probabilities /= probabilities.sum()
    noise = -np.log(np.maximum(rng.random(len(scores)), np.finfo(np.float32).tiny))
    return int(np.argmax(probabilities / noise))
