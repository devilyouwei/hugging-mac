"""YOLOv8 Seg App blueprint."""

from __future__ import annotations

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.instance_segmentation.manifest import INSTANCE_SEGMENTATION_MANIFEST
from hugging_mac_web.instance_segmentation.routes import create_router


class InstanceSegmentationBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return INSTANCE_SEGMENTATION_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> InstanceSegmentationBlueprint:
    return InstanceSegmentationBlueprint()
