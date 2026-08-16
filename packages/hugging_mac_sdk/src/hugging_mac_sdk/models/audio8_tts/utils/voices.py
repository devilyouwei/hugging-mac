"""Persistent reference-audio profiles for Audio8-TTS MLX."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from hugging_mac_sdk.schemas.transcription import AudioInput

_VOICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class VoiceStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, name: str, source: AudioInput, reference_text: str) -> tuple[Path, str]:
        profile = self._profile(name)
        profile.mkdir(parents=True, exist_ok=True)
        audio_path = profile / "reference.wav"
        if source.path is not None:
            shutil.copyfile(source.path, audio_path)
        else:
            assert source.data is not None
            audio_path.write_bytes(source.data)
        text = " ".join(reference_text.strip().split())
        if not text:
            raise ValueError("reference text must not be blank")
        (profile / "profile.json").write_text(
            json.dumps({"reference_text": text}, ensure_ascii=False), encoding="utf-8"
        )
        return audio_path, text

    def load(self, name: str) -> tuple[Path, str]:
        profile = self._profile(name)
        audio_path = profile / "reference.wav"
        metadata_path = profile / "profile.json"
        if not audio_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"voice profile does not exist: {name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        text = str(metadata.get("reference_text", "")).strip()
        if not text:
            raise ValueError(f"voice profile has no reference text: {name}")
        return audio_path, text

    def _profile(self, name: str) -> Path:
        if not _VOICE_NAME.fullmatch(name):
            raise ValueError(
                "voice name must contain only letters, numbers, dot, dash or underscore"
            )
        return self._root / name
