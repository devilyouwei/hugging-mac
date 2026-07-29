"""YOLOv8 Pose instances for PyTorch MPS and Core ML."""

from __future__ import annotations

import asyncio
import gc
import importlib
from abc import abstractmethod
from typing import Any, cast

from hugging_mac_sdk.capabilities import PoseEstimation
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.models._ultralytics_runtime import (
    class_name,
    decode_image,
    prediction_arguments,
    timings,
)
from hugging_mac_sdk.models.yolov8_pose.assets import YoloV8PoseAssetResolver
from hugging_mac_sdk.models.yolov8_pose.config import (
    YOLOV8_POSE_MODEL_ID,
    YOLOV8_POSE_REVISION,
    YoloV8PoseInstanceConfig,
)
from hugging_mac_sdk.schemas.detection import BoundingBox, DetectionRequest, ImageSize
from hugging_mac_sdk.schemas.pose import Keypoint, Pose, PoseEstimationResponse


class BaseYoloV8PoseInstance(BaseModelInstance):
    def __init__(self, config: YoloV8PoseInstanceConfig, assets: YoloV8PoseAssetResolver) -> None:
        super().__init__(
            model_id=YOLOV8_POSE_MODEL_ID,
            revision=YOLOV8_POSE_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device if config.runtime == "pytorch-mps" else config.compute_units,
        )
        self._config = config
        self._assets = assets
        self._model: Any = None
        self._inference_lock = asyncio.Lock()
        self.register_capability(PoseEstimation, cast(PoseEstimation, self))

    async def estimate_pose(self, request: DetectionRequest) -> PoseEstimationResponse:
        if self.state is not ModelState.READY or self._model is None:
            raise InferenceError("YOLOv8 Pose instance must be READY before estimate_pose")
        image = await asyncio.to_thread(decode_image, request)
        async with self._inference_lock:
            try:
                results = await asyncio.to_thread(
                    self._model.predict, image, **self._prediction_arguments(request)
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("YOLOv8 Pose inference failed", cause=error) from error
        if not results:
            raise InferenceError("YOLOv8 Pose returned no result")
        return _map_result(results[0], self)

    async def _load_model(self) -> None:
        if self._artifact_path is None:
            raise UnsupportedRuntimeError("YOLOv8 Pose artifact was not resolved")
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "YOLOv8 Pose runtime requires the yolo extra", cause=error
            ) from error
        self._model = await asyncio.to_thread(
            ultralytics.YOLO, str(self._artifact_path), task="pose"
        )

    async def _warmup(self) -> None:
        image = importlib.import_module("PIL.Image").new(
            "RGB", (self._config.imgsz, self._config.imgsz)
        )
        async with self._inference_lock:
            await asyncio.to_thread(
                self._model.predict,
                image,
                **self._prediction_arguments(DetectionRequest(image={"data": b"x"})),
            )

    async def _unload(self) -> None:
        self._model = None
        gc.collect()

    @abstractmethod
    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def execution_device(self) -> str: ...


class PyTorchMpsYoloV8PoseInstance(BaseYoloV8PoseInstance):
    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=(await self._assets.resolve_source()).path)

    async def _load(self) -> None:
        await self._load_model()
        torch = importlib.import_module("torch")
        device = self._config.device or "mps"
        if device == "mps" and not bool(torch.backends.mps.is_available()):
            if not self._config.allow_cpu_fallback:
                raise UnsupportedRuntimeError("PyTorch MPS is not available on this machine")
            device = "cpu"
        self._device = device
        self._set_runtime_context(device=device)

    async def _unload(self) -> None:
        loaded = self._model is not None
        await super()._unload()
        if loaded:
            torch = importlib.import_module("torch")
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]:
        return prediction_arguments(request, self._config.imgsz, device=self._device)

    @property
    def execution_device(self) -> str:
        return self._device


class CoreMlYoloV8PoseInstance(BaseYoloV8PoseInstance):
    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=(await self._assets.resolve_coreml()).path)

    async def _load(self) -> None:
        await self._load_model()

    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]:
        return prediction_arguments(request, self._config.imgsz)

    @property
    def execution_device(self) -> str:
        return self._config.compute_units


def _map_result(result: Any, instance: BaseYoloV8PoseInstance) -> PoseEstimationResponse:
    height, width = (int(value) for value in result.orig_shape)
    boxes = result.boxes
    keypoints = getattr(result, "keypoints", None)
    if boxes is None:
        poses: tuple[Pose, ...] = ()
    else:
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        points = keypoints.xy.cpu().tolist() if keypoints is not None else [[] for _ in coordinates]
        point_confidences = (
            keypoints.conf.cpu().tolist()
            if keypoints is not None and keypoints.conf is not None
            else [None] * len(coordinates)
        )
        poses = tuple(
            Pose(
                box=BoundingBox(
                    x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
                ),
                confidence=float(confidence),
                class_id=int(class_id),
                label=class_name(result.names, int(class_id)),
                keypoints=tuple(
                    Keypoint(
                        x=float(point[0]),
                        y=float(point[1]),
                        confidence=float(scores[index]) if scores is not None else None,
                    )
                    for index, point in enumerate(person_points)
                ),
            )
            for box, confidence, class_id, person_points, scores in zip(
                coordinates, confidences, classes, points, point_confidences, strict=True
            )
        )
    return PoseEstimationResponse(
        model_id=YOLOV8_POSE_MODEL_ID,
        instance_id=str(instance.instance_id),
        runtime=instance.info().runtime or "",
        device=instance.execution_device,
        image_size=ImageSize(width=width, height=height),
        poses=poses,
        timings=timings(result),
    )
