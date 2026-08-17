"""Runtime-independent RetinaFace instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from hugging_mac_sdk.capabilities import FaceDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.detection import (
    DetectionRequest,
    FaceDetectionResponse,
    ImageSize,
)

from .config import RETINAFACE_MODEL_ID, RETINAFACE_REVISION, RetinaFaceInstanceConfig
from .utils.postprocess import postprocess
from .utils.preprocess import prepare_image
from .utils.types import PreparedImage


class RetinaFaceEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer(self, prepared: PreparedImage, request: DetectionRequest) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class RetinaFaceInstance(BaseModelInstance):
    def __init__(self, config: RetinaFaceInstanceConfig, engine: RetinaFaceEngine) -> None:
        super().__init__(
            model_id=RETINAFACE_MODEL_ID,
            revision=RETINAFACE_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.compute_units,
        )
        self._config = config
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(FaceDetection, cast(FaceDetection, self))  # type: ignore[type-abstract]

    async def detect_faces(self, request: DetectionRequest) -> FaceDetectionResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("RetinaFace instance must be READY before detect_faces")
        prepared = await asyncio.to_thread(prepare_image, request, self._config.input_size)
        async with self._inference_lock:
            started = perf_counter()
            try:
                outputs = await self._engine.infer(prepared, request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("RetinaFace inference failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        faces, timings = await asyncio.to_thread(
            postprocess, dict(outputs), request, prepared, inference_ms
        )
        return FaceDetectionResponse(
            model_id=RETINAFACE_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            image_size=ImageSize(width=prepared.original_width, height=prepared.original_height),
            faces=faces,
            timings=timings,
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
