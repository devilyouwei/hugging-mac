"""Core ML runtime for the complete DeepFilterNet3 enhancement pipeline."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hugging_mac_sdk.runtime.base import RuntimeSession
from hugging_mac_sdk.runtime.coreml import CoreMLProvider

from .audio import PreparedEnhancementAudio
from .config import DeepFilterNet3InstanceConfig
from .dsp import DeepFilterNet3Dsp, EnhancementOutput
from .resources import DeepFilterNet3ResourceResolver


class CoreMlDeepFilterNet3Engine:
    runtime_name = "coreml"

    def __init__(
        self, config: DeepFilterNet3InstanceConfig, resources: DeepFilterNet3ResourceResolver
    ) -> None:
        self._config = config
        self._resources = resources
        self._session: RuntimeSession | None = None
        self._dsp: DeepFilterNet3Dsp | None = None
        self._inference_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deepfilternet3-coreml"
        )

    @property
    def device(self) -> str:
        return self._session.device if self._session is not None else (self._config.device or "all")

    async def resolve(self) -> Path:
        return (await self._resources.resolve()).path

    async def load(self, artifact: Path) -> None:
        self._session = await CoreMLProvider().create_session(
            artifact / "DeepFilterNet3.mlmodelc",
            device=self._config.device,
            options={},
            executor=self._executor,
        )
        self._dsp = await asyncio.to_thread(DeepFilterNet3Dsp, artifact / "auxiliary.npz")

    async def enhance(self, prepared: PreparedEnhancementAudio) -> EnhancementOutput:
        async with self._inference_lock:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._enhance_sync, prepared
            )

    def _enhance_sync(self, prepared: PreparedEnhancementAudio) -> EnhancementOutput:
        session, dsp = self._session, self._dsp
        if session is None or dsp is None:
            raise RuntimeError("DeepFilterNet3 Core ML engine is not loaded")
        return dsp.enhance(prepared.samples, lambda inputs: dict(session.run(inputs)))

    async def close(self) -> None:
        session, self._session = self._session, None
        self._dsp = None
        if session is not None:
            # Drop the native Core ML model on the same dedicated thread used
            # for construction and prediction.
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._close_session_sync, session
            )

    @staticmethod
    def _close_session_sync(session: RuntimeSession) -> None:
        asyncio.run(session.close())

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
