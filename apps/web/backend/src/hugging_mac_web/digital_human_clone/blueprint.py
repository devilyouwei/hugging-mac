"""Digital Human blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.digital_human_clone.config import DigitalHumanSettings
from hugging_mac_web.digital_human_clone.manifest import DIGITAL_HUMAN_MANIFEST
from hugging_mac_web.digital_human_clone.routes import create_router


class DigitalHumanBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return DIGITAL_HUMAN_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router(DigitalHumanSettings())


def create_blueprint() -> DigitalHumanBlueprint:
    return DigitalHumanBlueprint()
