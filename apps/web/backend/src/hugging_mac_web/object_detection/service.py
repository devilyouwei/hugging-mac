"""Object Detection business use case."""

from __future__ import annotations

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import ObjectDetection
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.object_detection.manifest import OBJECT_DETECTION_MANIFEST
from hugging_mac_web.object_detection.schemas import (
    DetectCommand,
    DetectionResult,
    ResourceStatusView,
    RuntimeChoice,
    VariantView,
)

YOLOV8_MODEL_ID = OBJECT_DETECTION_MANIFEST.required_models[0].model_id


class ObjectDetectionService:
    def __init__(self, context: PlatformContext) -> None:
        self._context = context

    async def resource_status(self, *, variant: str | None = None) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        status = await self._context.models.resources.status(
            YOLOV8_MODEL_ID,
            variant=selected_variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == YOLOV8_MODEL_ID),
            None,
        )
        return ResourceStatusView.from_sdk(
            status,
            default_runtime=model.default_runtime if model is not None else None,
            variants=tuple(
                VariantView(
                    name=item.name,
                    display_name=item.display_name,
                    description=item.description,
                    default=item.default,
                )
                for item in (model.variants if model is not None else ())
            ),
        )

    async def detect(
        self,
        command: DetectCommand,
        image: bytes,
        *,
        filename: str,
        content_type: str | None,
        suffix: str,
        image_metadata: dict[str, object],
        cache_input: bool = True,
    ) -> DetectionResult:
        input_cache_id: str | None = None
        if cache_input:
            cached = self._context.cache.put_bytes(
                "object-detection-inputs",
                image,
                suffix=suffix,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    **image_metadata,
                },
            )
            input_cache_id = cached.cache_id
        runtime = None if command.runtime is RuntimeChoice.AUTO else command.runtime.value
        selected_variant = self._resolve_variant(command.variant)
        async with await self._context.models.acquire(
            YOLOV8_MODEL_ID,
            variant=selected_variant,
            runtime=runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        ) as handle:
            # Protocol classes are the runtime capability keys. Mypy treats
            # Protocols as abstract even though no instance is constructed.
            detector = handle.require(ObjectDetection)  # type: ignore[type-abstract]
            response = await detector.detect(
                DetectionRequest(
                    image=ImageInput(data=image),
                    confidence=command.confidence,
                    iou_threshold=command.iou_threshold,
                    max_detections=command.max_detections,
                )
            )
        return DetectionResult.from_sdk(
            response,
            input_cache_id=input_cache_id,
            variant=selected_variant,
        )

    def _resolve_variant(self, variant: str | None) -> str:
        return self._context.models.registry.get(YOLOV8_MODEL_ID).manifest.get_variant(variant).name

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
