"""Object Detection App blueprint."""

from __future__ import annotations

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.object_detection.manifest import OBJECT_DETECTION_MANIFEST
from hugging_mac_web.object_detection.routes import create_router


class ObjectDetectionBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return OBJECT_DETECTION_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> ObjectDetectionBlueprint:
    return ObjectDetectionBlueprint()
