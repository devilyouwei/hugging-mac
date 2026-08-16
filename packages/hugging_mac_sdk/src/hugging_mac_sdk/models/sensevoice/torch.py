"""SenseVoiceSmall engine using PyTorch with Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.torch import TorchProvider

from .config import SenseVoiceSmallInstanceConfig
from .resources import SenseVoiceSmallResourceResolver
from .utils.types import (
    PreparedAudio,
    SenseVoiceEngineOutput,
    SenseVoiceInferenceOptions,
)


class TorchSenseVoiceSmallEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self,
        config: SenseVoiceSmallInstanceConfig,
        resources: SenseVoiceSmallResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._device: str = config.device or "mps"
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._artifact: Path | None = None
        self._dtype: Any | None = None

    @property
    def device(self) -> str:
        return self._device

    async def resolve(self) -> Path:
        artifact = await self._resources.resolve_source()
        await self._resources.resolve_tokenizer()
        return artifact.path

    async def load(self, artifact: Path) -> None:
        provider = TorchProvider()
        if not provider.is_available():
            raise UnsupportedRuntimeError(
                "PyTorch is not installed; install hugging-mac-sdk[asr]"
            )
        self._device = provider.resolve_device(
            self._config.device,
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        await asyncio.to_thread(self._load_sync, artifact)

    async def infer(
        self,
        prepared: PreparedAudio,
        options: SenseVoiceInferenceOptions,
    ) -> SenseVoiceEngineOutput:
        if self._model is None:
            raise RuntimeError("SenseVoiceSmall PyTorch engine is not loaded")
        return await asyncio.to_thread(self._infer_sync, prepared, options)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._tokenizer = None
        self._artifact = None
        if not had_model:
            return
        await asyncio.to_thread(gc.collect)
        if self._device == "mps" and self._torch is not None:
            empty_cache = getattr(getattr(self._torch, "mps", None), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()
        self._torch = None
        self._dtype = None

    def _load_sync(self, artifact: Path) -> None:
        torch = importlib.import_module("torch")
        sentencepiece = importlib.import_module("sentencepiece")
        yaml = importlib.import_module("yaml")
        from .utils.modeling import load_sensevoice_model

        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(
                self._resources.tokenizer_path
                / "chn_jpn_yue_eng_ko_spectok.bpe.model"
            )
        )
        config = yaml.safe_load((artifact / "config.yaml").read_text(encoding="utf-8"))
        dtype = self._resolve_dtype(torch)
        model = load_sensevoice_model(
            artifact,
            vocabulary_size=int(tokenizer.get_piece_size()),
            config=config,
            device=self._device,
            dtype=dtype,
        )
        self._torch = torch
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._artifact = artifact
        self._model = model

    def _infer_sync(
        self,
        prepared: PreparedAudio,
        options: SenseVoiceInferenceOptions,
    ) -> SenseVoiceEngineOutput:
        from .utils.frontend import extract_features
        from .utils.postprocess import parse_rich_transcript

        torch = self._torch
        model = self._model
        tokenizer = self._tokenizer
        artifact = self._artifact
        dtype = self._dtype
        if (
            torch is None
            or model is None
            or tokenizer is None
            or artifact is None
            or dtype is None
        ):
            raise RuntimeError("SenseVoiceSmall PyTorch engine is not loaded")
        features, lengths = extract_features(
            prepared,
            cmvn_path=artifact / "am.mvn",
            dither=self._config.fbank_dither,
        )
        features = features.to(device=self._device, dtype=dtype)
        lengths = lengths.to(device=self._device)
        with torch.inference_mode():
            token_ids = model.infer_tokens(
                features,
                lengths,
                language=options.language,
                use_itn=options.use_itn,
                ban_unknown_emotion=options.ban_unknown_emotion,
            )
        raw_text = tokenizer.decode_ids(token_ids.tolist())
        rich = parse_rich_transcript(raw_text)
        return SenseVoiceEngineOutput(
            text=rich.text,
            raw_text=rich.raw_text,
            languages=rich.languages,
            emotion=rich.emotion,
            events=rich.events,
            token_count=int(token_ids.numel()),
        )

    def _resolve_dtype(self, torch: Any) -> Any:
        requested = self._config.dtype
        if requested == "auto":
            requested = "float16" if self._device == "mps" else "float32"
        return getattr(torch, requested)
