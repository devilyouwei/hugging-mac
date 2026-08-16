"""Byte-level BPE tokenizer shipped with the aufklarer Core ML bundle."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path


class Qwen3CoreMlTokenizer:
    """Match speech-swift's small Qwen3 tokenizer without MLX tokenizer assets."""

    def __init__(self, vocab: dict[str, int], merge_ranks: dict[tuple[str, str], int]) -> None:
        self._vocab = vocab
        self._merge_ranks = merge_ranks
        self._cache: dict[str, tuple[str, ...]] = {}
        self._byte_encoder = self._build_byte_encoder()

    @classmethod
    def from_files(cls, vocab_path: Path, merges_path: Path) -> Qwen3CoreMlTokenizer:
        raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(raw_vocab, dict):
            raise ValueError("Qwen3-TTS Core ML vocab must be a token-to-id object")
        vocab = {str(token): int(token_id) for token, token_id in raw_vocab.items()}
        merge_ranks: dict[tuple[str, str], int] = {}
        for rank, line in enumerate(merges_path.read_text(encoding="utf-8").splitlines()):
            if not line or line.startswith("#"):
                continue
            pieces = line.split(" ")
            if len(pieces) == 2:
                merge_ranks[(pieces[0], pieces[1])] = rank
        if not vocab or not merge_ranks:
            raise ValueError("Qwen3-TTS Core ML tokenizer files are empty")
        return cls(vocab, merge_ranks)

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for word in self._pretokenize(text):
            for token in self._bpe(word):
                token_id = self._vocab.get(token)
                if token_id is None:
                    raise ValueError(f"Qwen3-TTS tokenizer produced an unknown token: {token!r}")
                encoded.append(token_id)
        return encoded

    def _pretokenize(self, text: str) -> list[str]:
        words: list[str] = []
        current = ""
        for character in text:
            if character in {" ", "\n", "\t"}:
                if current:
                    words.append(self._byte_encode(current))
                current = character
            else:
                current += character
        if current:
            words.append(self._byte_encode(current))
        return words

    def _byte_encode(self, text: str) -> str:
        return "".join(self._byte_encoder[value] for value in text.encode("utf-8"))

    def _bpe(self, word: str) -> tuple[str, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        pieces = list(word)
        while len(pieces) > 1:
            pair = min(
                pairwise(pieces),
                key=lambda item: self._merge_ranks.get(item, 2**63 - 1),
            )
            if pair not in self._merge_ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == pair:
                    merged.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged
        result = tuple(pieces)
        self._cache[word] = result
        return result

    @staticmethod
    def _build_byte_encoder() -> dict[int, str]:
        values = list(range(ord("!"), ord("~") + 1))
        values += list(range(0xA1, 0xAC + 1))
        values += list(range(0xAE, 0xFF + 1))
        unicode_values = values.copy()
        extra = 0
        for value in range(256):
            if value not in values:
                values.append(value)
                unicode_values.append(256 + extra)
                extra += 1
        return dict(zip(values, (chr(value) for value in unicode_values), strict=True))
