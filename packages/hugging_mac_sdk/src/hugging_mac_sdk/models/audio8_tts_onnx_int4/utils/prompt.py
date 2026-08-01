"""Audio8 prompt construction for the exported ONNX graphs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer  # type: ignore[import-untyped]


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def format_reference_text(text: str) -> str:
    value = clean_text(text)
    return value if re.search(r"<\|speaker:\d+\|>", value) else f"<|speaker:0|>{value}"


class PromptBuilder:
    def __init__(self, tokenizer_dir: Path, semantic_begin_id: int, num_codebooks: int):
        self._tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self._semantic_begin_id = int(semantic_begin_id)
        self._num_codebooks = int(num_codebooks)

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def build(
        self,
        target_text: str,
        reference_text: str,
        reference_codes: np.ndarray,
    ) -> np.ndarray:
        codes = np.asarray(reference_codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != self._num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"reference codes must have shape [{self._num_codebooks}, T>0], got {codes.shape}"
            )
        prefix_parts = (
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            format_reference_text(reference_text),
            "\n\nSpeech:\n",
        )
        suffix_parts = (
            "<|im_end|>\n",
            "<|im_start|>user\n",
            clean_text(target_text),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        )
        prefix = [token for part in prefix_parts for token in self._encode(part)]
        suffix = [token for part in suffix_parts for token in self._encode(part)]
        semantic_ids = (codes[0] + self._semantic_begin_id).tolist()
        first_row = np.asarray(prefix + semantic_ids + suffix, dtype=np.int64)
        values = np.zeros((self._num_codebooks + 1, first_row.size), dtype=np.int64)
        values[0] = first_row
        begin = len(prefix)
        values[1:, begin : begin + codes.shape[1]] = codes
        return values[np.newaxis]
