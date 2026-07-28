"""YOLOv8 object detection instances for PyTorch MPS and Core ML."""

from __future__ import annotations

import asyncio
import gc
import importlib
import io
from abc import abstractmethod
from pathlib import Path
from typing import Any, cast

from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError, UnsupportedRuntimeError
from hugging_mac_sdk.models.yolov8.assets import YoloV8AssetResolver
from hugging_mac_sdk.models.yolov8.config import YoloV8InstanceConfig
from hugging_mac_sdk.models.yolov8.config import (
    YOLOV8_MODEL_ID,
    YOLOV8_MODEL_REVISION,
)
from hugging_mac_sdk.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionRequest,
    DetectionResponse,
    DetectionTimings,
    ImageSize,
)


class BaseYoloV8Instance(BaseModelInstance):
    runtime_name: str

    def __init__(
        self,
        config: YoloV8InstanceConfig,
        assets: YoloV8AssetResolver,
    ) -> None:
        super().__init__(
            model_id=YOLOV8_MODEL_ID,
            revision=YOLOV8_MODEL_REVISION,
            runtime=config.runtime,
            device=config.device if config.runtime == "pytorch-mps" else config.compute_units,
        )
        self._config = config
        self._assets = assets
        self._artifact_path: Path | None = None
        self._model: Any = None
        self._inference_lock = asyncio.Lock()
        self.register_capability(ObjectDetection, cast(ObjectDetection, self))

    async def detect(self, request: DetectionRequest) -> DetectionResponse:
        if self.state is not ModelState.READY or self._model is None:
            raise InferenceError("YOLOv8 instance must be READY before detect")

        image = await asyncio.to_thread(_decode_image, request)
        arguments = self._prediction_arguments(request)
        async with self._inference_lock:
            try:
                results = await asyncio.to_thread(self._model.predict, image, **arguments)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("YOLOv8 detection failed", cause=error) from error
        if not results:
            raise InferenceError("YOLOv8 returned no result")
        return _map_result(results[0], self)

    async def _load(self) -> None:
        if self._artifact_path is None:
            raise UnsupportedRuntimeError("YOLOv8 artifact was not resolved")
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise UnsupportedRuntimeError(
                "YOLOv8 runtime requires: uv sync --package hugging-mac-sdk --extra yolo",
                cause=error,
            ) from error
        self._model = await asyncio.to_thread(
            ultralytics.YOLO,
            str(self._artifact_path),
            task="detect",
        )

    async def _warmup(self) -> None:
        pillow_image = _pillow_image_module()
        image = pillow_image.new("RGB", (self._config.imgsz, self._config.imgsz))
        async with self._inference_lock:
            await asyncio.to_thread(
                self._model.predict,
                image,
                **self._prediction_arguments(
                    DetectionRequest(image={"data": b"x"}, confidence=0.25)
                ),
            )

    async def _unload(self) -> None:
        self._model = None
        gc.collect()

    @abstractmethod
    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def execution_device(self) -> str: ...


class PyTorchMpsYoloV8Instance(BaseYoloV8Instance):
    runtime_name = "pytorch-mps"

    async def _resolve(self) -> None:
        self._artifact_path = (await self._assets.resolve_source()).path
        self._set_runtime_context(artifact_path=self._artifact_path)

    async def _load(self) -> None:
        await super()._load()
        torch = importlib.import_module("torch")
        mps_available = bool(torch.backends.mps.is_available())
        requested = self._config.device or "mps"
        if requested == "mps" and not mps_available:
            if not self._config.allow_cpu_fallback:
                raise UnsupportedRuntimeError("PyTorch MPS is not available on this machine")
            requested = "cpu"
        self._device = requested
        self._set_runtime_context(device=requested)

    async def _unload(self) -> None:
        had_loaded_model = self._model is not None
        await super()._unload()
        if not had_loaded_model:
            return
        torch = importlib.import_module("torch")
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]:
        return _common_prediction_arguments(request, self._config.imgsz) | {
            "device": self._device
        }

    @property
    def execution_device(self) -> str:
        return self._device


class CoreMlYoloV8Instance(BaseYoloV8Instance):
    runtime_name = "coreml"

    async def _resolve(self) -> None:
        self._artifact_path = (await self._assets.resolve_coreml()).path
        self._set_runtime_context(artifact_path=self._artifact_path)

    def _prediction_arguments(self, request: DetectionRequest) -> dict[str, Any]:
        # Ultralytics detects .mlpackage and dispatches through Core ML. Core ML's
        # default compute unit is ALL; a device string here would refer to torch.
        return _common_prediction_arguments(request, self._config.imgsz)

    @property
    def execution_device(self) -> str:
        return self._config.compute_units


def _common_prediction_arguments(
    request: DetectionRequest,
    image_size: int,
) -> dict[str, Any]:
    return {
        "conf": request.confidence,
        "iou": request.iou_threshold,
        "max_det": request.max_detections,
        "classes": list(request.classes) if request.classes is not None else None,
        "imgsz": image_size,
        "verbose": False,
    }


def _decode_image(request: DetectionRequest) -> Any:
    if request.image.path is not None:
        path = request.image.path.expanduser()
        if not path.is_file():
            raise InferenceError(f"Image file does not exist: {path}")
        return str(path)
    assert request.image.data is not None
    try:
        image_module = _pillow_image_module()
        image = image_module.open(io.BytesIO(request.image.data))
        image.load()
        return image.convert("RGB")
    except Exception as error:
        raise InferenceError("Image bytes could not be decoded", cause=error) from error


def _pillow_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as error:
        raise UnsupportedRuntimeError("Image decoding requires Pillow", cause=error) from error


def _map_result(result: Any, instance: BaseYoloV8Instance) -> DetectionResponse:
    height, width = (int(value) for value in result.orig_shape)
    names = result.names
    boxes = result.boxes
    if boxes is None:
        detections: tuple[Detection, ...] = ()
    else:
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        detections = tuple(
            Detection(
                box=BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ),
                confidence=float(confidence),
                class_id=int(class_id),
                label=_class_name(names, int(class_id)),
            )
            for box, confidence, class_id in zip(
                coordinates,
                confidences,
                classes,
                strict=True,
            )
        )
    speed = getattr(result, "speed", {}) or {}
    return DetectionResponse(
        model_id="ultralytics/yolov8n",
        instance_id=str(instance.instance_id),
        runtime=instance.runtime_name,
        device=instance.execution_device,
        image_size=ImageSize(width=width, height=height),
        detections=detections,
        timings=DetectionTimings(
            preprocess_ms=_optional_float(speed.get("preprocess")),
            inference_ms=_optional_float(speed.get("inference")),
            postprocess_ms=_optional_float(speed.get("postprocess")),
        ),
    )


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
