"""YOLOv8 Seg instances for PyTorch MPS and Core ML."""

from __future__ import annotations

import asyncio
import gc
import importlib
from abc import abstractmethod
from typing import Any, cast

from hugging_mac_sdk.capabilities import InstanceSegmentation
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.models._ultralytics_runtime import (
    class_name,
    decode_image,
    prediction_arguments,
    timings,
)
from hugging_mac_sdk.models.yolov8_seg.assets import YoloV8SegAssetResolver
from hugging_mac_sdk.models.yolov8_seg.config import (
    YOLOV8_SEG_MODEL_ID,
    YOLOV8_SEG_REVISION,
    YoloV8SegInstanceConfig,
)
from hugging_mac_sdk.schemas.detection import BoundingBox, DetectionRequest, ImageSize
from hugging_mac_sdk.schemas.segmentation import PolygonPoint, Segmentation, SegmentationResponse


class BaseYoloV8SegInstance(BaseModelInstance):
    def __init__(self, config: YoloV8SegInstanceConfig, assets: YoloV8SegAssetResolver) -> None:
        super().__init__(
            model_id=YOLOV8_SEG_MODEL_ID,
            revision=YOLOV8_SEG_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.device if config.runtime == "pytorch-mps" else config.compute_units,
        )
        self._config = config
        self._assets = assets
        self._model: Any = None
        self._inference_lock = asyncio.Lock()
        self.register_capability(InstanceSegmentation, cast(InstanceSegmentation, self))

    async def segment(self, request: DetectionRequest) -> SegmentationResponse:
        if self.state is not ModelState.READY or self._model is None:
            raise InferenceError("YOLOv8 Seg instance must be READY before segment")
        image = await asyncio.to_thread(decode_image, request)
        async with self._inference_lock:
            try:
                results = await asyncio.to_thread(
                    self._model.predict, image, **self._prediction_arguments(request)
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("YOLOv8 Seg inference failed", cause=error) from error
        if not results:
            raise InferenceError("YOLOv8 Seg returned no result")
        return _map_result(results[0], self)

    async def _load_model(self) -> None:
        if self._artifact_path is None:
            raise UnsupportedRuntimeError("YOLOv8 Seg artifact was not resolved")
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "YOLOv8 Seg runtime requires the yolo extra", cause=error
            ) from error
        self._model = await asyncio.to_thread(
            ultralytics.YOLO, str(self._artifact_path), task="segment"
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


class PyTorchMpsYoloV8SegInstance(BaseYoloV8SegInstance):
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


class CoreMlYoloV8SegInstance(BaseYoloV8SegInstance):
    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=(await self._assets.resolve_coreml()).path)

    async def _load(self) -> None:
        await self._load_model()

    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]:
        return prediction_arguments(request, self._config.imgsz)

    @property
    def execution_device(self) -> str:
        return self._config.compute_units


def _map_result(result: Any, instance: BaseYoloV8SegInstance) -> SegmentationResponse:
    height, width = (int(value) for value in result.orig_shape)
    boxes = result.boxes
    masks = getattr(result, "masks", None)
    if boxes is None:
        segments: tuple[Segmentation, ...] = ()
    else:
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        # Ultralytics exposes one contour per instance as a NumPy array.
        # NumPy arrays intentionally do not support truth-value testing, so
        # use their length explicitly before constructing the SDK polygon.
        contours = masks.xy if masks is not None else [() for _ in coordinates]
        segments = tuple(
            Segmentation(
                box=BoundingBox(
                    x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
                ),
                confidence=float(confidence),
                class_id=int(class_id),
                label=class_name(result.names, int(class_id)),
                polygons=(
                    tuple(
                        PolygonPoint(x=float(point[0]), y=float(point[1]))
                        for point in contour
                    ),
                )
                if len(contour) > 0
                else (),
            )
            for box, confidence, class_id, contour in zip(
                coordinates, confidences, classes, contours, strict=True
            )
        )
    return SegmentationResponse(
        model_id=YOLOV8_SEG_MODEL_ID,
        instance_id=str(instance.instance_id),
        runtime=instance.info().runtime or "",
        device=instance.execution_device,
        image_size=ImageSize(width=width, height=height),
        segments=segments,
        timings=timings(result),
    )
