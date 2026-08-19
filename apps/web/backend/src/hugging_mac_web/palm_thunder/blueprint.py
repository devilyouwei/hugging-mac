"""Palm Thunder blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest

from .manifest import PALM_THUNDER_MANIFEST
from .routes import create_router


class PalmThunderBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return PALM_THUNDER_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> PalmThunderBlueprint:
    return PalmThunderBlueprint()
