"""Model catalog and live instance endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status
from hugging_mac_sdk import (
    ArtifactFormat,
    InstanceSnapshot,
    ModelResourceStatus,
    ModelSummary,
    ResourceNotFoundError,
    ReusePolicy,
    UnloadResult,
    UnsupportedCapabilityError,
)
from pydantic import BaseModel, ConfigDict

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.time_util import utc_now


class LoadModelCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str = "auto"
    variant: str | None = None
    device: str | None = None
    warmup: bool = False


class ConvertModelCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_format: ArtifactFormat
    variant: str | None = None


def create_models_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/catalog", tags=["models"])

    @router.get("/models", response_model=ApiResponse[list[ModelSummary]])
    async def list_models(
        context: ContextDependency,
    ) -> ApiResponse[list[ModelSummary]]:
        snapshot = await context.models.catalog.snapshot()
        return ApiResponse(
            data=list(snapshot.models),
            meta=ResponseMeta(generated_at=snapshot.generated_at),
        )

    @router.get(
        "/models/{model_id:path}/resources",
        response_model=ApiResponse[ModelResourceStatus],
    )
    async def model_resources(
        model_id: str,
        context: ContextDependency,
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ModelResourceStatus]:
        resources = await context.models.resources.status(
            model_id,
            variant=variant,
            options={"model_home": context.settings.model_home},
        )
        return ApiResponse(
            data=resources,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/models/{model_id:path}/resources/download",
        response_model=ApiResponse[ModelResourceStatus],
    )
    async def download_model_resources(
        model_id: str,
        context: ContextDependency,
        variant: str | None = Query(default=None),
        overwrite: bool = Query(default=False),
    ) -> ApiResponse[ModelResourceStatus]:
        selected_variant = _resolve_variant(context, model_id, variant)
        await _ensure_resources_mutable(
            context,
            model_id,
            variant=selected_variant,
        )
        resources = await context.models.resources.download_source(
            model_id,
            variant=selected_variant,
            options={"model_home": context.settings.model_home},
            overwrite=overwrite,
        )
        return ApiResponse(
            data=resources,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.delete(
        "/models/{model_id:path}/resources",
        response_model=ApiResponse[ModelResourceStatus],
    )
    async def delete_model_resources(
        model_id: str,
        context: ContextDependency,
        runtime: str | None = Query(default=None),
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ModelResourceStatus]:
        selected_variant = _resolve_variant(context, model_id, variant)
        await _ensure_resources_mutable(
            context,
            model_id,
            runtime=runtime,
            variant=selected_variant,
        )
        resources = await context.models.resources.delete(
            model_id,
            variant=selected_variant,
            runtime=runtime,
            options={"model_home": context.settings.model_home},
        )
        return ApiResponse(
            data=resources,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/models/{model_id:path}/resources/convert",
        response_model=ApiResponse[ModelResourceStatus],
    )
    async def convert_model_resources(
        model_id: str,
        command: ConvertModelCommand,
        context: ContextDependency,
    ) -> ApiResponse[ModelResourceStatus]:
        selected_variant = _resolve_variant(context, model_id, command.variant)
        await _ensure_resources_mutable(
            context,
            model_id,
            variant=selected_variant,
        )
        resources = await context.models.resources.convert(
            model_id,
            command.target_format,
            variant=selected_variant,
            options={"model_home": context.settings.model_home},
            overwrite=True,
        )
        return ApiResponse(
            data=resources,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/models/{model_id:path}/instances",
        response_model=ApiResponse[InstanceSnapshot],
        status_code=status.HTTP_201_CREATED,
    )
    async def load_model(
        model_id: str,
        command: LoadModelCommand,
        context: ContextDependency,
    ) -> ApiResponse[InstanceSnapshot]:
        instance = await context.models.instances.load(
            model_id,
            variant=command.variant,
            runtime=command.runtime,
            device=command.device,
            options={"model_home": context.settings.model_home},
            reuse=ReusePolicy.DEDICATED,
            warmup=command.warmup,
        )
        snapshot = await context.models.instances.snapshot(str(instance.instance_id))
        return ApiResponse(
            data=snapshot,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.delete(
        "/instances/{instance_id}",
        response_model=ApiResponse[UnloadResult],
    )
    async def unload_model(
        instance_id: str,
        context: ContextDependency,
    ) -> ApiResponse[UnloadResult]:
        result = await context.models.instances.unload_with_metrics(instance_id)
        if result is None:
            raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
        return ApiResponse(
            data=result,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router


def _resolve_variant(
    context: ContextDependency,
    model_id: str,
    variant: str | None,
) -> str:
    return context.models.registry.get(model_id).manifest.get_variant(variant).name


async def _ensure_resources_mutable(
    context: ContextDependency,
    model_id: str,
    *,
    runtime: str | None = None,
    variant: str | None = None,
) -> None:
    active = tuple(
        snapshot
        for snapshot in await context.models.instances.snapshots()
        if snapshot.model_id == model_id
        and (runtime is None or snapshot.runtime == runtime)
        and (variant is None or snapshot.variant == variant)
    )
    if active:
        raise UnsupportedCapabilityError(
            "Unload all matching model instances before changing model files",
            details={
                "model_id": model_id,
                "runtime": runtime,
                "variant": variant,
                "instance_count": len(active),
            },
        )
