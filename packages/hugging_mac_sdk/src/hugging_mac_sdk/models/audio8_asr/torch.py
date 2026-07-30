"""Audio8-ASR engine using PyTorch with automatic Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.transcription import TranscriptionRequest

from .config import Audio8AsrInstanceConfig
from .resources import Audio8AsrResourceResolver
from .utils.types import AsrEngineOutput, PreparedAudio


class TorchAudio8AsrEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self,
        config: Audio8AsrInstanceConfig,
        resources: Audio8AsrResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self._device: str = config.device or "mps"
        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._feature_extractor: Any | None = None
        self._dtype: Any | None = None
        self._audio_token = "<|audio|>"
        self._merge_factor = 4

    @property
    def device(self) -> str:
        return self._device

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        provider = TorchProvider()
        if not provider.is_available():
            raise UnsupportedRuntimeError("PyTorch is not installed; install hugging-mac-sdk[asr]")
        self._device = provider.resolve_device(
            self._config.device,
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        await asyncio.to_thread(self._load_sync, artifact)

    async def infer(
        self,
        prepared: PreparedAudio,
        request: TranscriptionRequest,
    ) -> AsrEngineOutput:
        if self._model is None:
            raise RuntimeError("Audio8-ASR PyTorch engine is not loaded")
        return await asyncio.to_thread(self._infer_sync, prepared, request)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._tokenizer = None
        self._feature_extractor = None
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
        transformers = importlib.import_module("transformers")
        from .utils.modeling import load_audio8_asr_model

        dtype = self._resolve_dtype(torch)
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(
            artifact,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
            artifact,
            local_files_only=True,
        )
        model = load_audio8_asr_model(
            artifact,
            device=self._device,
            dtype=dtype,
        )
        self._torch = torch
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._feature_extractor = feature_extractor
        self._model = model

    def _infer_sync(
        self,
        prepared: PreparedAudio,
        request: TranscriptionRequest,
    ) -> AsrEngineOutput:
        torch = self._torch
        model = self._model
        tokenizer = self._tokenizer
        feature_extractor = self._feature_extractor
        dtype = self._dtype
        if (
            torch is None
            or model is None
            or tokenizer is None
            or feature_extractor is None
            or dtype is None
        ):
            raise RuntimeError("Audio8-ASR PyTorch engine is not loaded")

        features = feature_extractor(
            prepared.samples,
            sampling_rate=prepared.sample_rate,
            return_tensors="pt",
            return_attention_mask=False,
            padding="longest",
            max_length=int(self._config.max_audio_seconds * prepared.sample_rate),
        )["input_features"]
        hop_length = int(getattr(feature_extractor, "hop_length", 160))
        mel_frames = int(prepared.samples.shape[0]) // max(hop_length, 1)
        audio_tokens = max(((mel_frames + 1) // 2) // self._merge_factor, 1)
        prompt = (
            "<|user|><|begin_of_audio|>"
            + self._audio_token * audio_tokens
            + "<|end_of_audio|>"
            + request.prompt
            + "<|assistant|>"
        )
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        input_features = features.to(device=self._device, dtype=dtype)
        eos_token_id = int(tokenizer.eos_token_id)
        with torch.inference_mode():
            generated = model.generate_transcript(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=eos_token_id,
            )
        text = tokenizer.batch_decode(
            generated.detach().cpu(),
            skip_special_tokens=True,
        )[0]
        return AsrEngineOutput(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=int(generated.shape[1]),
        )

    def _resolve_dtype(self, torch: Any) -> Any:
        requested = self._config.dtype
        if requested == "auto":
            requested = "float16" if self._device == "mps" else "float32"
        return getattr(torch, requested)
