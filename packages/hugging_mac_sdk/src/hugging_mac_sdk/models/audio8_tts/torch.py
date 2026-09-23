"""Audio8-TTS engine using the stable PyTorch CPU path."""

from __future__ import annotations

import asyncio
import gc
import importlib
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from hugging_mac_sdk.errors import UnsupportedRuntimeError
from hugging_mac_sdk.resources.views import merged_directory_view
from hugging_mac_sdk.runtime.torch import TorchProvider
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Audio8TtsInstanceConfig
from .resources import Audio8TtsResourceResolver
from .utils.types import TtsEngineOutput


def _install_falcon_h1_cache_compatibility(transformers: Any) -> None:
    """Expose the Transformers 4 Falcon-H1 cache API to Audio8 on v5."""

    version = getattr(transformers, "__version__", "")
    try:
        major_version = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return
    if major_version < 5:
        return

    falcon_h1 = importlib.import_module("transformers.models.falcon_h1.modeling_falcon_h1")
    legacy_name = "FalconHybridMambaAttentionDynamicCache"
    if hasattr(falcon_h1, legacy_name):
        return

    dynamic_cache = transformers.DynamicCache

    class FalconHybridMambaAttentionDynamicCache(dynamic_cache):  # type: ignore[misc, valid-type]
        """Accept the removed v4 constructor and use the v5 lazy cache."""

        def __init__(
            self,
            config: Any,
            batch_size: int,
            dtype: Any,
            devices: list[Any] | None = None,
        ) -> None:
            del batch_size, dtype, devices
            super().__init__(config=config)

    FalconHybridMambaAttentionDynamicCache.__module__ = falcon_h1.__name__
    setattr(falcon_h1, legacy_name, FalconHybridMambaAttentionDynamicCache)


def _precompute_rope(torch: Any, length: int, head_dim: int, base: float) -> Any:
    """Build the pinned ArkTTS rotary buffer outside Transformers' init guard."""

    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim))
    phases = torch.outer(torch.arange(length), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _restore_rope_buffers(torch: Any, model: Any) -> None:
    """Restore non-persistent buffers skipped by Transformers 5 fast init."""

    config = model.config
    device = next(model.parameters()).device
    model.freqs_cis = _precompute_rope(
        torch,
        int(config.max_seq_len),
        int(config.head_dim),
        float(config.rope_base),
    ).to(device)
    model.fast_freqs_cis = _precompute_rope(
        torch,
        int(config.num_codebooks),
        int(config.fast_head_dim),
        float(config.rope_base),
    ).to(device)
    if not bool(torch.isfinite(model.freqs_cis).all()) or not bool(
        torch.isfinite(model.fast_freqs_cis).all()
    ):
        raise RuntimeError("Audio8-TTS rotary buffers contain non-finite values")


def _restore_falcon_h1_buffers(transformers: Any, model: Any) -> None:
    """Restore Falcon-H1 MuP buffers skipped by Transformers 5 fast init."""

    if not str(getattr(transformers, "__version__", "")).startswith("5."):
        return
    slow = getattr(model, "slow", None)
    layers = getattr(slow, "layers", None)
    if slow is None or layers is None:
        return
    falcon_h1 = importlib.import_module("transformers.models.falcon_h1.modeling_falcon_h1")
    compute_mup_vector = getattr(falcon_h1, "compute_mup_vector", None)
    if compute_mup_vector is None:
        return
    vector = compute_mup_vector(slow.config)
    for layer in layers:
        target = layer.mamba.mup_vector
        layer.mamba.mup_vector = vector.clone().to(device=target.device, dtype=target.dtype)


class TorchAudio8TtsEngine:
    def __init__(
        self,
        config: Audio8TtsInstanceConfig,
        resources: Audio8TtsResourceResolver,
    ) -> None:
        self._config = config
        self._resources = resources
        self.runtime_name = config.runtime
        self._device: str = config.device
        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._dtype: Any | None = None
        self._model_view: AbstractContextManager[Path] | None = None
        self._artifact: Path | None = None

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
            raise UnsupportedRuntimeError("PyTorch is not installed; install hugging-mac-sdk[tts]")
        self._device = provider.resolve_device(
            self._config.device,
            allow_cpu_fallback=True,
        )
        self._artifact = artifact
        try:
            await asyncio.to_thread(self._load_sync, artifact)
        except (RuntimeError, NotImplementedError):
            if self._device != "mps":
                raise
            self._device = "cpu"
            await asyncio.to_thread(self._load_sync, artifact)

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        if self._model is None:
            raise RuntimeError("Audio8-TTS PyTorch engine is not loaded")
        try:
            return await asyncio.to_thread(self._infer_sync, request)
        except (RuntimeError, NotImplementedError):
            if self._device != "mps" or self._artifact is None:
                raise
            await asyncio.to_thread(self._reload_on_cpu_sync)
            return await asyncio.to_thread(self._infer_sync, request)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        had_model = self._model is not None
        self._model = None
        self._processor = None
        model_view, self._model_view = self._model_view, None
        if model_view is not None:
            model_view.__exit__(None, None, None)
        if not had_model:
            return
        gc.collect()
        if self._device == "mps" and self._torch is not None:
            self._torch.mps.empty_cache()
        self._torch = None
        self._dtype = None
        self._artifact = None

    def _reload_on_cpu_sync(self) -> None:
        assert self._artifact is not None
        artifact = self._artifact
        self._close_sync()
        self._device = "cpu"
        self._artifact = artifact
        self._load_sync(artifact)

    def _load_sync(self, artifact: Path) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        _install_falcon_h1_cache_compatibility(transformers)
        dtype = self._resolve_dtype(torch)
        model_view = merged_directory_view(artifact, (self._resources.tokenizer_path,))
        path = model_view.__enter__()
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = (
                transformers.AutoModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=dtype,
                )
                .eval()
                .to(self._device)
            )
            # Transformers 5 guards tensor initialization while constructing custom
            # models. ArkTTS creates its non-persistent RoPE buffers in __init__, so
            # that guard leaves them as uninitialized memory (often NaN). Rebuild
            # them after loading; otherwise generation never emits EOS and the codec
            # receives invalid frames, producing a fixed-length noise waveform.
            _restore_rope_buffers(torch, model)
            _restore_falcon_h1_buffers(transformers, model)
        except BaseException:
            model_view.__exit__(None, None, None)
            raise
        self._torch = torch
        self._dtype = dtype
        self._processor = processor
        self._model = model
        self._model_view = model_view

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
            waveform = waveforms[0, :sample_count].detach().float().cpu().contiguous().numpy()
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
            requested = "float32"
        return getattr(torch, requested)
