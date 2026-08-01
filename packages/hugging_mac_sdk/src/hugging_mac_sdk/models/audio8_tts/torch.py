"""Audio8-TTS engine using PyTorch with Apple MPS selection."""

from __future__ import annotations

import asyncio
import gc
import importlib
import tempfile
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Audio8TtsInstanceConfig
from .resources import Audio8TtsResourceResolver
from .utils.types import TtsEngineOutput


class TorchAudio8TtsEngine:
    runtime_name = "pytorch-mps"

    def __init__(
        self,
        config: Audio8TtsInstanceConfig,
        resources: Audio8TtsResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        # Audio8's DualAR sampler is numerically unstable on the current MPS
        # backend: it can fail to emit EOS and generate invalid codec frames.
        # CPU FP32 is the reliable automatic path; callers may still opt into
        # MPS explicitly while that upstream/runtime limitation is investigated.
        self._device: str = config.device or "cpu"
        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._dtype: Any | None = None

    @property
    def device(self) -> str:
        return self._device

    async def resolve(self) -> Path:
        return (await self._resources.resolve_source()).path

    async def load(self, artifact: Path) -> None:
        provider = TorchProvider()
        if not provider.is_available():
            raise UnsupportedRuntimeError(
                "PyTorch is not installed; install hugging-mac-sdk[tts]"
            )
        self._device = provider.resolve_device(
            self._config.device or "cpu",
            allow_cpu_fallback=self._config.allow_cpu_fallback,
        )
        await asyncio.to_thread(self._load_sync, artifact)

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        if self._model is None:
            raise RuntimeError("Audio8-TTS PyTorch engine is not loaded")
        return await asyncio.to_thread(self._infer_sync, request)

    async def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._processor = None
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
        dtype = self._resolve_dtype(torch)
        processor = transformers.AutoProcessor.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = transformers.AutoModel.from_pretrained(
            artifact,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).eval().to(self._device)
        self._torch = torch
        self._dtype = dtype
        self._processor = processor
        self._model = model

    def _infer_sync(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        torch = self._torch
        processor = self._processor
        model = self._model
        if torch is None or processor is None or model is None:
            raise RuntimeError("Audio8-TTS PyTorch engine is not loaded")

        reference_path: Path | None = None
        temporary_path: str | None = None
        if request.reference_audio is not None:
            if request.reference_audio.path is not None:
                reference_path = request.reference_audio.path
            else:
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
                    assert request.reference_audio.data is not None
                    handle.write(request.reference_audio.data)
                    temporary_path = handle.name
                    reference_path = Path(handle.name)
        try:
            inputs = processor(
                text=[request.text],
                reference_audio=([reference_path] if reference_path is not None else None),
                reference_text=(
                    [request.reference_text] if request.reference_text is not None else None
                ),
                return_tensors="pt",
            )
            inputs = {
                name: value.to(self._device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=request.do_sample,
                    return_dict_in_generate=True,
                )
                waveforms, lengths = model.decode_audio(output.codes)
            sample_count = int(lengths[0])
            waveform = (
                waveforms[0, :sample_count]
                .detach()
                .float()
                .cpu()
                .contiguous()
                .numpy()
            )
            sample_rate = int(model.config.codec_sample_rate)
            generated_tokens = int(output.codes.shape[-1])
            return TtsEngineOutput(
                audio=waveform.astype("<f4", copy=False).tobytes(),
                sample_rate=sample_rate,
                duration_seconds=sample_count / sample_rate,
                generated_tokens=generated_tokens,
            )
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    def _resolve_dtype(self, torch: Any) -> Any:
        requested = self._config.dtype
        if requested == "auto":
            # Audio8's released checkpoint is BF16.  Casting its autoregressive
            # logits to FP16 on MPS can prevent EOS generation and produce
            # invalid codec frames, so preserve BF16 unless the caller opts in
            # to another precision explicitly.
            requested = "bfloat16" if self._device == "mps" else "float32"
        return getattr(torch, requested)
