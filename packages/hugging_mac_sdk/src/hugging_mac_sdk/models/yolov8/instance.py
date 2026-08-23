"""Runtime-independent YOLOv8 object-detection instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    ImageSize,
)
from hugging_mac_sdk.schemas.manifest import ModelManifest

from .config import YoloV8InstanceConfig
from .utils.postprocess import postprocess
from .utils.preprocess import prepare_image
from .utils.types import PreparedImage


class YoloV8Engine(Protocol):
    """Model-private runtime engine composed into the public instance."""

    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def infer(
        self,
        prepared: PreparedImage,
        request: DetectionRequest,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class YoloV8Instance(BaseModelInstance):
    """ObjectDetection capability composed with one selected runtime engine."""

    def __init__(
        self, config: YoloV8InstanceConfig, engine: YoloV8Engine, manifest: ModelManifest
    ) -> None:
        assert manifest.model_id is not None
        super().__init__(
            model_id=manifest.model_id,
            revision=manifest.revision,
            variant=config.variant,
            runtime=config.runtime,
            device=(config.compute_units if config.runtime == "coreml" else config.device),
        )
        self._config = config
        self._manifest_model_id = manifest.model_id
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(
            ObjectDetection,  # type: ignore[type-abstract]
            cast(ObjectDetection, self),
        )

    async def detect(self, request: DetectionRequest) -> DetectionResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("YOLOv8 instance must be READY before detect")
        prepared = await asyncio.to_thread(prepare_image, request, self._config.imgsz)
        async with self._inference_lock:
            started = perf_counter()
            try:
                outputs = await self._engine.infer(prepared, request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("YOLOv8 detection failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        detections, result_timings = await asyncio.to_thread(
            postprocess,
            dict(outputs),
            request,
            prepared,
            inference_ms,
        )
        return DetectionResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            image_size=ImageSize(
                width=prepared.original_width,
                height=prepared.original_height,
            ),
            detections=detections,
            timings=result_timings,
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
