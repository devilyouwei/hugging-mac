"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from hugging_mac_web.context import PlatformContext


def get_context(request: Request) -> PlatformContext:
    return cast(PlatformContext, request.app.state.context)


ContextDependency = Annotated[PlatformContext, Depends(get_context)]
