"""Runtime-independent MediaPipe Hand detector and optional landmarker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast

from hugging_mac_sdk.capabilities import HandDetection
from hugging_mac_sdk.core.instance import BaseModelInstance, ModelState
from hugging_mac_sdk.errors import InferenceError
from hugging_mac_sdk.schemas.detection import BoundingBox, DetectionTimings, ImageSize
from hugging_mac_sdk.schemas.hand import (
    HandDetectionRequest,
    HandDetectionResponse,
    HandLandmark,
    HandResult,
)

from .config import (
    MEDIAPIPE_HAND_DETECTION_MODEL_ID,
    MEDIAPIPE_HAND_DETECTION_REVISION,
    MediaPipeHandDetectionInstanceConfig,
)
from .utils.postprocess import decode_landmarks, decode_palms
from .utils.preprocess import prepare_image, prepare_landmark_crop
from .utils.types import PreparedImage, PreparedLandmarkCrop


class MediaPipeHandDetectionEngine(Protocol):
    runtime_name: str

    @property
    def device(self) -> str: ...
    async def resolve(self) -> Path: ...
    async def load(self, artifact: Path) -> None: ...
    async def infer_detector(self, prepared: PreparedImage) -> dict[str, Any]: ...
    async def infer_landmarks(self, prepared: PreparedLandmarkCrop) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class MediaPipeHandDetectionInstance(BaseModelInstance):
    def __init__(
        self,
        config: MediaPipeHandDetectionInstanceConfig,
        engine: MediaPipeHandDetectionEngine,
    ) -> None:
        super().__init__(
            model_id=MEDIAPIPE_HAND_DETECTION_MODEL_ID,
            revision=MEDIAPIPE_HAND_DETECTION_REVISION,
            variant=config.variant,
            runtime=config.runtime,
            device=config.compute_units if config.runtime == "coreml" else config.device,
        )
        self._config = config
        self._engine = engine
        self._inference_lock = asyncio.Lock()
        self.register_capability(HandDetection, cast(HandDetection, self))  # type: ignore[type-abstract]

    async def detect_hands(self, request: HandDetectionRequest) -> HandDetectionResponse:
        if self.state is not ModelState.READY:
            raise InferenceError("MediaPipe Hand Detection must be READY before detect_hands")
        prepared = await asyncio.to_thread(prepare_image, request, self._config.input_size)
        inference_ms = 0.0
        postprocess_ms = 0.0
        preprocess_ms = prepared.preprocess_ms
        async with self._inference_lock:
            started = perf_counter()
            try:
                detector_outputs = await self._engine.infer_detector(prepared)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise InferenceError("MediaPipe palm detection failed", cause=error) from error
            inference_ms += (perf_counter() - started) * 1000
            palms, elapsed = await asyncio.to_thread(
                decode_palms, detector_outputs, request, prepared
            )
            postprocess_ms += elapsed
            hands: list[HandResult] = []
            for palm in palms:
                landmarks: tuple[HandLandmark, ...] = ()
                landmark_confidence: float | None = None
                handedness: Literal["left", "right"] | None = None
                handedness_confidence: float | None = None
                if request.include_landmarks:
                    crop = await asyncio.to_thread(
                        prepare_landmark_crop, prepared, palm, self._config.input_size
                    )
                    preprocess_ms += crop.preprocess_ms
                    started = perf_counter()
                    try:
                        landmark_outputs = await self._engine.infer_landmarks(crop)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        raise InferenceError(
                            "MediaPipe hand landmark detection failed", cause=error
                        ) from error
                    inference_ms += (perf_counter() - started) * 1000
                    (
                        landmarks,
                        landmark_confidence,
                        handedness,
                        handedness_confidence,
                        elapsed,
                    ) = await asyncio.to_thread(
                        decode_landmarks,
                        landmark_outputs,
                        crop,
                        minimum_confidence=request.landmark_confidence,
                        image_width=prepared.original_width,
                        image_height=prepared.original_height,
                        input_mirrored=request.input_mirrored,
                    )
                    postprocess_ms += elapsed
                x1, y1, x2, y2 = palm.box
                hands.append(
                    HandResult(
                        box=BoundingBox(
                            x1=min(max(x1, 0.0), float(prepared.original_width)),
                            y1=min(max(y1, 0.0), float(prepared.original_height)),
                            x2=min(max(x2, 0.0), float(prepared.original_width)),
                            y2=min(max(y2, 0.0), float(prepared.original_height)),
                        ),
                        confidence=palm.confidence,
                        landmarks=landmarks,
                        landmark_confidence=landmark_confidence,
                        handedness=handedness,
                        handedness_confidence=handedness_confidence,
                    )
                )
        return HandDetectionResponse(
            model_id=MEDIAPIPE_HAND_DETECTION_MODEL_ID,
            instance_id=str(self.instance_id),
            runtime=self._engine.runtime_name,
            device=self._engine.device,
            image_size=ImageSize(width=prepared.original_width, height=prepared.original_height),
            hands=tuple(hands),
            landmarks_enabled=request.include_landmarks,
            timings=DetectionTimings(
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
            ),
        )

    async def _resolve(self) -> None:
        self._set_runtime_context(artifact_path=await self._engine.resolve())

    async def _load(self) -> None:
        assert self._artifact_path is not None
        await self._engine.load(self._artifact_path)
        self._set_runtime_context(device=self._engine.device)

    async def _unload(self) -> None:
        await self._engine.close()
