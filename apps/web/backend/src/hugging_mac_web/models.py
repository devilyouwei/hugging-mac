"""Model catalog and live instance endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from hugging_mac_sdk import ModelSummary

from hugging_mac_web.dependencies import ContextDependency
from hugging_mac_web.schemas import ApiResponse, ResponseMeta


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

    return router
