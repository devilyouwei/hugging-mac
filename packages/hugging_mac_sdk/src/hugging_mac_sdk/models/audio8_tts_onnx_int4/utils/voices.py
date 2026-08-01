"""Validated local voice profiles for Audio8 ONNX inference."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


class VoiceStore:
    def __init__(self, root: Path, num_codebooks: int, fingerprint: str):
        self.root = root.expanduser().resolve()
        self._num_codebooks = int(num_codebooks)
        self._fingerprint = fingerprint

    @staticmethod
    def validate_name(name: str) -> str:
        value = name.strip()
        if not value or len(value) > 64 or value in {".", ".."} or Path(value).name != value:
            raise ValueError("voice must be one path component with at most 64 characters")
        return value

    def load(self, name: str) -> tuple[np.ndarray, str]:
        voice_name = self.validate_name(name)
        directory = self.root / voice_name
        meta_path = directory / "meta.json"
        codes_path = directory / "codes.npy"
        if not meta_path.is_file() or not codes_path.is_file():
            raise KeyError(f"voice profile not found: {voice_name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_fingerprint") != self._fingerprint:
            raise ValueError(f"voice profile is incompatible with this model: {voice_name}")
        codes = np.load(codes_path, allow_pickle=False).astype(np.int64, copy=False)
        self._validate_codes(codes)
        reference_text = str(meta.get("reference_text", "")).strip()
        if not reference_text:
            raise ValueError(f"voice profile has no reference text: {voice_name}")
        return codes, reference_text

    def save(self, name: str, codes: np.ndarray, reference_text: str) -> None:
        voice_name = self.validate_name(name)
        values = np.asarray(codes, dtype=np.int64)
        self._validate_codes(values)
        text = " ".join(reference_text.strip().split())
        if not text:
            raise ValueError("reference text must not be blank")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / voice_name
        temporary = Path(tempfile.mkdtemp(prefix=f".{voice_name}.", dir=self.root))
        try:
            np.save(temporary / "codes.npy", values.astype(np.uint16))
            meta = {
                "name": voice_name,
                "reference_text": text,
                "shape": list(values.shape),
                "dtype": "uint16",
                "model_fingerprint": self._fingerprint,
                "source_kind": "sdk_reference_audio",
            }
            (temporary / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _validate_codes(self, codes: np.ndarray) -> None:
        if codes.ndim != 2 or codes.shape[0] != self._num_codebooks or codes.shape[1] == 0:
            raise ValueError(f"invalid voice profile codes: {codes.shape}")
