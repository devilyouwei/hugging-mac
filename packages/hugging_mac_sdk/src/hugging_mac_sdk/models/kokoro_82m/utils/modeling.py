"""Shared Kokoro loading, phonemization, and voice helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def load_kokoro_model(
    artifact: Path,
    *,
    device: str,
    disable_complex: bool,
) -> Any:
    KModel = importlib.import_module("kokoro").KModel

    return (
        KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(artifact / "config.json"),
            model=str(artifact / "kokoro-v1_0.pth"),
            disable_complex=disable_complex,
        )
        .eval()
        .to(device)
    )


def phonemize_chunks(text: str, language: str) -> list[str]:
    KPipeline = importlib.import_module("kokoro").KPipeline

    pipeline = KPipeline(
        lang_code=language,
        repo_id="hexgrad/Kokoro-82M",
        model=False,
    )
    return [
        result.phonemes
        for result in pipeline(text, voice=None)
        if result.phonemes
    ]


def load_voice(torch: Any, artifact: Path, voice: str, phoneme_count: int) -> Any:
    path = artifact / "voices" / f"{voice.removesuffix('.pt')}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Kokoro voice is not downloaded: {voice}")
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if pack.ndim == 3:
        index = min(max(phoneme_count - 1, 0), int(pack.shape[0]) - 1)
        pack = pack[index]
    if pack.ndim == 1:
        pack = pack.unsqueeze(0)
    return pack


def phonemes_to_ids(model: Any, torch: Any, phonemes: str, device: str) -> Any:
    ids = [model.vocab.get(symbol) for symbol in phonemes]
    filtered = [value for value in ids if value is not None]
    if len(filtered) + 2 > int(model.context_length):
        raise ValueError("Kokoro phoneme chunk exceeds the model context")
    return torch.tensor([[0, *filtered, 0]], dtype=torch.long, device=device)
