"""YOLOv8 Seg business use case."""

from __future__ import annotations

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import InstanceSegmentation
from hugging_mac_sdk.models.yolov8_seg.config import YOLOV8_SEG_MODEL_ID
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.instance_segmentation.schemas import (
    ResourceStatusView,
    RuntimeChoice,
    SegmentationCommand,
    SegmentationResult,
    VariantView,
)


class InstanceSegmentationService:
    def __init__(self, context: PlatformContext) -> None:
        self._context = context

    async def resource_status(self, *, variant: str | None = None) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        status = await self._context.models.resources.status(
            YOLOV8_SEG_MODEL_ID,
            variant=selected_variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == YOLOV8_SEG_MODEL_ID),
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

    async def segment(
        self,
        command: SegmentationCommand,
        image: bytes,
        *,
        filename: str,
        content_type: str | None,
        suffix: str,
        image_metadata: dict[str, object],
        cache_input: bool = True,
    ) -> SegmentationResult:
        input_cache_id: str | None = None
        if cache_input:
            cached = self._context.cache.put_bytes(
                "instance-segmentation-inputs",
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
            YOLOV8_SEG_MODEL_ID,
            variant=selected_variant,
            runtime=runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        ) as handle:
            segmenter = handle.require(InstanceSegmentation)  # type: ignore[type-abstract]
            response = await segmenter.segment(
                DetectionRequest(
                    image=ImageInput(data=image),
                    confidence=command.confidence,
                    iou_threshold=command.iou_threshold,
                    max_detections=command.max_detections,
                )
            )
        return SegmentationResult.from_sdk(
            response,
            input_cache_id=input_cache_id,
            variant=selected_variant,
        )

    def _resolve_variant(self, variant: str | None) -> str:
        return (
            self._context.models.registry.get(YOLOV8_SEG_MODEL_ID)
            .manifest.get_variant(variant)
            .name
        )

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
