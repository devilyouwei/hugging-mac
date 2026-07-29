"""YOLOv8 Pose business use case."""

from __future__ import annotations

from hugging_mac_sdk import ReusePolicy
from hugging_mac_sdk.capabilities import PoseEstimation
from hugging_mac_sdk.schemas.conversion import ArtifactFormat
from hugging_mac_sdk.schemas.detection import DetectionRequest, ImageInput

from hugging_mac_web.context import PlatformContext
from hugging_mac_web.pose_estimation.config import PoseEstimationSettings
from hugging_mac_web.pose_estimation.schemas import (
    PoseCommand,
    PoseResult,
    ResourceStatusView,
    RuntimeChoice,
    VariantView,
)


class PoseEstimationService:
    def __init__(
        self,
        context: PlatformContext,
        settings: PoseEstimationSettings,
    ) -> None:
        self._context = context
        self._settings = settings

    async def resource_status(self, *, variant: str | None = None) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        status = await self._context.models.resources.status(
            self._settings.model_id,
            variant=selected_variant,
            options=self._model_options,
        )
        catalog = await self._context.models.catalog.snapshot()
        model = next(
            (item for item in catalog.models if item.model_id == self._settings.model_id),
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

    async def download_source(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
    ) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        await self._context.models.resources.download_source(
            self._settings.model_id,
            variant=selected_variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status(variant=selected_variant)

    async def convert_coreml(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
    ) -> ResourceStatusView:
        selected_variant = self._resolve_variant(variant)
        await self._context.models.resources.convert(
            self._settings.model_id,
            ArtifactFormat.COREML,
            variant=selected_variant,
            options=self._model_options,
            overwrite=overwrite,
        )
        return await self.resource_status(variant=selected_variant)

    async def estimate(
        self,
        command: PoseCommand,
        image: bytes,
        *,
        filename: str,
        content_type: str | None,
        suffix: str,
        image_metadata: dict[str, object],
        cache_input: bool = True,
    ) -> PoseResult:
        input_cache_id: str | None = None
        if cache_input:
            cached = self._context.cache.put_bytes(
                "pose-estimation-inputs",
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
            self._settings.model_id,
            variant=selected_variant,
            runtime=runtime,
            options=self._model_options,
            reuse=ReusePolicy.SHARED,
        ) as handle:
            estimator = handle.require(PoseEstimation)  # type: ignore[type-abstract]
            response = await estimator.estimate_pose(
                DetectionRequest(
                    image=ImageInput(data=image),
                    confidence=command.confidence,
                    iou_threshold=command.iou_threshold,
                    max_detections=command.max_detections,
                )
            )
        return PoseResult.from_sdk(
            response,
            input_cache_id=input_cache_id,
            variant=selected_variant,
        )

    def _resolve_variant(self, variant: str | None) -> str:
        return self._context.models.registry.get(
            self._settings.model_id
        ).manifest.get_variant(variant or self._settings.model_variant).name

    @property
    def _model_options(self) -> dict[str, object]:
        return {"model_home": self._context.settings.model_home}
