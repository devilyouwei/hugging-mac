"""System and health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from hugging_mac_web.schemas import ApiResponse, ResponseMeta
from hugging_mac_web.shared.utils.system_util import system_snapshot
from hugging_mac_web.shared.utils.time_util import utc_now


def create_system_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/system", tags=["system"])

    @router.get("/health", response_model=ApiResponse[dict[str, str]])
    async def health() -> ApiResponse[dict[str, str]]:
        return ApiResponse(
            data={"status": "healthy"},
            meta=ResponseMeta(generated_at=utc_now()),
        )

    @router.get("/info", response_model=ApiResponse[dict[str, Any]])
    async def info() -> ApiResponse[dict[str, Any]]:
        return ApiResponse(
            data=system_snapshot(),
            meta=ResponseMeta(generated_at=utc_now()),
        )

    return router
