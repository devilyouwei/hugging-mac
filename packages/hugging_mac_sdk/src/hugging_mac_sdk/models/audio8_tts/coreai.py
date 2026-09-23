"""Hybrid Audio8 pipeline: Core AI fast AR/decoder, PyTorch CPU slow AR/encoder."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from pathlib import Path
from typing import Any

from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.errors import ResourceNotFoundError
from hugging_mac_sdk.runtime.coreai import CoreAIProvider, CoreAISession
from hugging_mac_sdk.schemas.artifact import ModelArtifact
from hugging_mac_sdk.schemas.speech_synthesis import SpeechSynthesisRequest

from .config import Audio8TtsCoreAIInstanceConfig, Audio8TtsInstanceConfig
from .resources import Audio8TtsResourceResolver
from .torch import TorchAudio8TtsEngine
from .utils.types import CoreAILayout, TtsEngineOutput


async def _settled_thread(function: Any, *args: Any) -> Any:
    """Keep native work inside its lifecycle lock even when the caller cancels."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


class CoreAIAudio8TtsEngine:
    runtime_name: str = "coreai"

    def __init__(
        self,
        config: Audio8TtsCoreAIInstanceConfig,
        artifact: ModelArtifact,
        resources: Audio8TtsResourceResolver,
        layout: CoreAILayout,
    ) -> None:
        self._config = config
        self._artifact = artifact
        self._resources = resources
        self._layout = layout
        self._cpu = TorchAudio8TtsEngine(
            Audio8TtsInstanceConfig(variant=config.variant, model_home=config.model_home), resources
        )
        self._fast: CoreAISession | None = None
        self._codec: CoreAISession | None = None
        self._keys: Any = None
        self._values: Any = None
        self._lock = asyncio.Lock()

    @property
    def device(self) -> str:
        return self._config.device

    async def resolve(self) -> Path:
        path = self._config.source_path or self._artifact.resolve(self._config.model_home)
        if not artifact_available(self._artifact, path):
            raise ResourceNotFoundError(
                "Audio8 Core AI artifact is incomplete; download or convert it first"
            )
        await self._resources.resolve_source()
        await self._resources.resolve_tokenizer()
        return path

    async def load(self, artifact: Path) -> None:
        async with self._lock:
            try:
                self._fast = await CoreAIProvider().create_session(
                    artifact / self._layout.fast_graph, device=self.device, options={}
                )
                self._codec = await CoreAIProvider().create_session(
                    artifact / self._layout.codec_graph, device=self.device, options={}
                )
                await _settled_thread(
                    self._cpu._load_sync, artifact / self._layout.pytorch_directory
                )
                self._install_hooks()
            except BaseException:
                await self._close()
                raise

    def _install_hooks(self) -> None:
        model = self._cpu._model
        if model is None:
            raise RuntimeError("Audio8 CPU stages failed to load")
        model._fast_step = self._fast_step
        model.decode_audio = self._decode_audio
        # These weights now live in Core AI; sampling and embeddings stay in PyTorch.
        torch = self._cpu._torch
        assert torch is not None
        model.fast_layers = torch.nn.ModuleList()
        model.fast_norm = torch.nn.Identity()
        model.fast_output = torch.nn.Identity()

    def _fast_step(self, hidden: Any, position: int) -> Any:
        if self._fast is None or self._cpu._model is None:
            raise RuntimeError("Audio8 Core AI fast decoder is not loaded")
        np = importlib.import_module("numpy")
        config = self._cpu._model.config
        if hidden.shape[0] != 1 or not 0 <= position < config.num_codebooks:
            raise ValueError(
                "Audio8 Core AI fast decoding requires batch size 1 and a valid position"
            )
        if position == 0:
            shape = (
                config.n_fast_layer,
                1,
                config.fast_n_local_heads,
                config.num_codebooks,
                config.fast_head_dim,
            )
            self._keys = np.zeros(shape, dtype=np.float32)
            self._values = np.zeros(shape, dtype=np.float32)
        if self._keys is None:
            raise RuntimeError("Fast AR frame must start at position zero")
        outputs = self._fast.run(
            {
                "hidden": hidden.detach().float().cpu().numpy(),
                "position": np.asarray([position], dtype=np.int32),
                "keys": self._keys,
                "values": self._values,
            }
        )
        self._keys, self._values = outputs["new_keys"], outputs["new_values"]
        torch = self._cpu._torch
        assert torch is not None
        return torch.from_numpy(outputs["logits"])

    def _decode_audio(self, codes: Any) -> tuple[Any, Any]:
        if self._codec is None or self._cpu._model is None:
            raise RuntimeError("Audio8 Core AI codec is not loaded")
        torch = self._cpu._torch
        assert torch is not None
        config = self._cpu._model.config
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != config.num_codebooks:
            raise ValueError("Audio8 Core AI codec requires one batch of complete codebooks")
        length = int((codes[0] >= 0).all(dim=0).sum().item())
        if length == 0:
            return torch.empty(1, 0), torch.tensor([0])
        if length > self._layout.codec_max_frames:
            raise ValueError("Generated audio exceeds the converted codec frame limit")
        np = importlib.import_module("numpy")
        result = self._codec.run({"codes": codes[:, :, :length].cpu().numpy().astype(np.int32)})
        waveform = torch.from_numpy(result["waveform"]).reshape(1, -1)
        return waveform, torch.tensor([waveform.shape[-1]])

    async def infer(self, request: SpeechSynthesisRequest) -> TtsEngineOutput:
        async with self._lock:
            try:
                return await _settled_thread(self._cpu._infer_sync, request)  # type: ignore[no-any-return]
            finally:
                self._keys = self._values = None
                if self._cpu._model is not None:
                    self._cpu._model.__dict__["_slow_cache"] = None

    async def _close(self) -> None:
        try:
            await _settled_thread(self._cpu._close_sync)
        finally:
            sessions = (self._fast, self._codec)
            self._fast = self._codec = None
            self._keys = self._values = None
            results = await asyncio.gather(
                *(session.close() for session in sessions if session is not None),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    async def close(self) -> None:
        async with self._lock:
            await self._close()
