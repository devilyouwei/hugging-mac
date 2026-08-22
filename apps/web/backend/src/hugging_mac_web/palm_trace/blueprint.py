"""Palm Trace blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest

from .manifest import PALM_TRACE_MANIFEST
from .routes import create_router


class PalmTraceBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return PALM_TRACE_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> PalmTraceBlueprint:
    return PalmTraceBlueprint()
