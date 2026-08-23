"""Model catalog and live instance endpoints."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Query, status
from hugging_mac_sdk import (
    InstanceSnapshot,
    ModelArtifact,
    ModelDefinition,
    ModelResourceStatus,
    ModelStructure,
    ModelSummary,
    ResourceNotFoundError,
    ReusePolicy,
    UnloadResult,
    UnsupportedCapabilityError,
)
from hugging_mac_sdk.core.resources import artifact_available
from hugging_mac_sdk.resources.downloader import ResourceDownloader
from hugging_mac_sdk.resources.hashing import directory_size
from hugging_mac_sdk.schemas.resources import ResourceSource
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


class ArtifactCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str | None = None
    runtime: str | None = None
    artifact_id: str


class ConvertArtifactCommand(ArtifactCommand):
    model_config = ConfigDict(frozen=True)

    overwrite: bool = False


class ArtifactInventoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str | None
    runtime: str | None
    artifact_id: str
    format: str
    required_shares: tuple[str, ...] = ()
    convertible: bool
    source: ResourceSource | None = None
    shared: bool = False
    available: bool
    size_bytes: int | None = None


class ModelInventory(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    revision: str
    artifacts: tuple[ArtifactInventoryItem, ...]


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

    @router.get(
        "/models/{model_id:path}/inventory",
        response_model=ApiResponse[ModelInventory],
    )
    async def model_inventory(
        model_id: str,
        context: ContextDependency,
    ) -> ApiResponse[ModelInventory]:
        return ApiResponse(
            data=_inventory(context, model_id),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.get(
        "/models/{model_id:path}/structure",
        response_model=ApiResponse[ModelStructure],
    )
    async def model_structure(
        model_id: str,
        context: ContextDependency,
        runtime: str = Query(),
        artifact_id: str = Query(),
        variant: str | None = Query(default=None),
    ) -> ApiResponse[ModelStructure]:
        structure = await context.models.inspection.structure(
            model_id,
            variant=variant,
            runtime=runtime,
            artifact_id=artifact_id,
            model_home=context.settings.model_home,
        )
        return ApiResponse(
            data=structure,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.post(
        "/models/{model_id:path}/resources/download-one",
        response_model=ApiResponse[ModelInventory],
    )
    async def download_one_resource(
        model_id: str,
        command: ArtifactCommand,
        context: ContextDependency,
        overwrite: bool = Query(default=False),
    ) -> ApiResponse[ModelInventory]:
        definition = context.models.registry.get(model_id)
        artifact = _find_artifact(definition, command)
        source = _artifact_source(artifact)
        if source is None:
            raise UnsupportedCapabilityError(
                "Artifact is not downloadable; it must be converted",
                details=command.model_dump(),
            )
        destination = artifact.resolve(context.settings.model_home)
        available = artifact_available(artifact, destination)
        if available and not overwrite:
            return ApiResponse(
                data=_inventory(context, model_id),
                meta=ResponseMeta(generated_at=utc_now()),
            )
        if artifact.shared:
            await _ensure_shared_resources_mutable(
                context,
                definition,
                variant=command.variant,
                overwrite=overwrite,
            )
            await context.models.resources.download_shared_artifact(
                model_id,
                artifact.artifact_id,
                options={"model_home": context.settings.model_home},
                overwrite=overwrite,
            )
            return ApiResponse(
                data=_inventory(context, model_id),
                meta=ResponseMeta(generated_at=utc_now()),
            )
        await _ensure_resources_mutable(
            context,
            model_id,
            runtime=artifact.runtime,
            variant=artifact.variant,
        )
        await _ensure_shared_resources_mutable(
            context,
            definition,
            variant=artifact.variant,
            runtime=artifact.runtime,
            artifact_id=artifact.artifact_id,
            overwrite=False,
        )
        await context.models.resources.ensure_shared_artifacts(
            model_id,
            variant=artifact.variant,
            runtime=artifact.runtime,
            artifact_id=artifact.artifact_id,
            options={"model_home": context.settings.model_home},
            overwrite=False,
        )
        await ResourceDownloader().download(source, destination, overwrite=overwrite)
        return ApiResponse(
            data=_inventory(context, model_id), meta=ResponseMeta(generated_at=utc_now())
        )

    @router.delete(
        "/models/{model_id:path}/artifacts",
        response_model=ApiResponse[ModelInventory],
    )
    async def delete_artifact(
        model_id: str,
        command: ArtifactCommand,
        context: ContextDependency,
    ) -> ApiResponse[ModelInventory]:
        await _ensure_resources_mutable(
            context, model_id, runtime=command.runtime, variant=command.variant
        )
        definition = context.models.registry.get(model_id)
        artifact = _find_artifact(definition, command)
        if artifact.shared:
            await _ensure_resources_mutable(context, model_id)
        path = artifact.resolve(context.settings.model_home)
        await asyncio.to_thread(_delete_path, path, context.settings.model_home)
        return ApiResponse(
            data=_inventory(context, model_id), meta=ResponseMeta(generated_at=utc_now())
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
        command: ConvertArtifactCommand,
        context: ContextDependency,
    ) -> ApiResponse[ModelResourceStatus]:
        definition = context.models.registry.get(model_id)
        artifact = _find_artifact(definition, command)
        if artifact.shared or not artifact.convert:
            raise UnsupportedCapabilityError(
                "Conversion is not enabled for this artifact",
                details={"model_id": model_id, **command.model_dump()},
            )
        assert artifact.variant is not None
        assert artifact.runtime is not None
        await _ensure_shared_resources_mutable(
            context,
            definition,
            variant=artifact.variant,
            runtime=artifact.runtime,
            artifact_id=artifact.artifact_id,
            overwrite=False,
        )
        await _ensure_resources_mutable(
            context,
            model_id,
            runtime=artifact.runtime,
            variant=artifact.variant,
        )
        resources = await context.models.resources.convert(
            model_id,
            artifact.format,
            variant=artifact.variant,
            options={"model_home": context.settings.model_home},
            overwrite=command.overwrite,
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
            reuse=ReusePolicy.SHARED,
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
        force: bool = Query(default=False),
    ) -> ApiResponse[UnloadResult]:
        result = await context.models.instances.unload_with_metrics(instance_id, force=force)
        if result is None:
            raise ResourceNotFoundError(f"Model instance not found: {instance_id}")
        return ApiResponse(
            data=result,
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router


def _inventory(context: ContextDependency, model_id: str) -> ModelInventory:
    definition = context.models.registry.get(model_id)
    artifacts: list[ArtifactInventoryItem] = []
    for artifact in definition.artifacts:
        path = artifact.resolve(context.settings.model_home)
        available = artifact_available(artifact, path)
        size = (
            (directory_size(path) if path.is_dir() else path.stat().st_size) if available else None
        )
        artifacts.append(
            ArtifactInventoryItem(
                variant=artifact.variant,
                runtime=artifact.runtime,
                artifact_id=artifact.artifact_id,
                format=artifact.format.value,
                required_shares=artifact.required_shares,
                convertible=artifact.convert,
                source=_artifact_source(artifact),
                shared=artifact.shared,
                available=available,
                size_bytes=size,
            )
        )
    return ModelInventory(
        model_id=model_id,
        revision=definition.manifest.revision,
        artifacts=tuple(artifacts),
    )


def _artifact_source(artifact: ModelArtifact) -> ResourceSource | None:
    return artifact.source


def _find_artifact(definition: ModelDefinition, command: ArtifactCommand) -> ModelArtifact:
    for artifact in definition.artifacts:
        if artifact.artifact_id != command.artifact_id:
            continue
        if artifact.shared:
            if command.variant is None:
                return artifact
            required_ids = {
                item.artifact_id
                for item in definition.required_shared_artifacts(variant=command.variant)
            }
            if artifact.artifact_id in required_ids:
                return artifact
            continue
        if (artifact.variant, artifact.runtime) == (command.variant, command.runtime):
            return artifact
    raise ResourceNotFoundError("Artifact is not declared", details=command.model_dump())


def _delete_path(path: Path, model_home: Path) -> None:
    root = model_home.expanduser().resolve(strict=False)
    target = path.expanduser().resolve(strict=False)
    if target == root or not target.is_relative_to(root):
        raise UnsupportedCapabilityError("Refusing to delete an artifact outside model storage")
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    elif target.is_dir():
        shutil.rmtree(target)


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


async def _ensure_shared_resources_mutable(
    context: ContextDependency,
    definition: ModelDefinition,
    *,
    variant: str | None = None,
    runtime: str | None = None,
    artifact_id: str | None = None,
    target_format: str | None = None,
    overwrite: bool,
) -> None:
    shared_artifacts = (
        definition.required_shared_artifacts(
            variant=variant,
            runtime=runtime,
            artifact_id=artifact_id,
            target_format=target_format,
        )
        if variant is not None
        else definition.shared_artifacts
    )
    if not shared_artifacts:
        return
    needs_mutation = overwrite or any(
        not _declared_artifact_available(artifact, context.settings.model_home)
        for artifact in shared_artifacts
    )
    if needs_mutation:
        await _ensure_resources_mutable(context, definition.manifest.model_id)


def _declared_artifact_available(artifact: ModelArtifact, model_home: Path) -> bool:
    path = artifact.resolve(model_home)
    return artifact_available(artifact, path)
