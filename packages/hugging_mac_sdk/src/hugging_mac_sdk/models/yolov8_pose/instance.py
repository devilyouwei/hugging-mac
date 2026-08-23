"""Runtime-independent YOLOv8 Pose instance."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from hugging_mac_sdk.capabilities import PoseEstimation
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.detection import ImageSize
from hugging_mac_sdk.schemas.manifest import ModelManifest
from hugging_mac_sdk.schemas.pose import PoseEstimationResponse, PoseRequest

from .config import YoloV8PoseInstanceConfig
from .utils.postprocess import postprocess
from .utils.preprocess import prepare_image
from .utils.types import PreparedImage


class YoloV8PoseEngine(Protocol):
    """Model-private runtime engine composed into the public instance."""

    runtime_name: str

    @property
    def device(self) -> str: ...

    async def resolve(self) -> Path: ...

    async def load(self, artifact: Path) -> None: ...

    async def infer(
        self,
        prepared: PreparedImage,
        request: PoseRequest,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class YoloV8PoseInstance(BaseModelInstance):
    """PoseEstimation capability composed with one selected runtime engine."""

    def __init__(
        self, config: YoloV8PoseInstanceConfig, engine: YoloV8PoseEngine, manifest: ModelManifest
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
            PoseEstimation,  # type: ignore[type-abstract]
            cast(PoseEstimation, self),
        )

    async def estimate_pose(self, request: PoseRequest) -> PoseEstimationResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("YOLOv8 Pose instance must be READY before estimate_pose")
        prepared = await asyncio.to_thread(prepare_image, request, self._config.imgsz)
        async with self._inference_lock:
            started = perf_counter()
            try:
                outputs = await self._engine.infer(prepared, request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("YOLOv8 Pose inference failed", cause=error) from error
            inference_ms = (perf_counter() - started) * 1000
        poses, result_timings = await asyncio.to_thread(
            postprocess,
            dict(outputs),
            request,
            prepared,
            inference_ms,
        )
        return PoseEstimationResponse(
            model_id=self._manifest_model_id,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            image_size=ImageSize(
                width=prepared.original_width,
                height=prepared.original_height,
            ),
            poses=poses,
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
