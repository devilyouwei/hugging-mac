"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends
from starlette.requests import HTTPConnection

from hugging_mac_web.context import PlatformContext


def get_context(connection: HTTPConnection) -> PlatformContext:
    return cast(PlatformContext, connection.app.state.context)


ContextDependency = Annotated[PlatformContext, Depends(get_context)]
