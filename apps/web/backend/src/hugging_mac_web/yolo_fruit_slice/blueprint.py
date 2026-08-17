"""YOLO Fruit Slice blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.yolo_fruit_slice.manifest import YOLO_FRUIT_SLICE_MANIFEST
from hugging_mac_web.yolo_fruit_slice.routes import create_router


class YoloFruitSliceBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return YOLO_FRUIT_SLICE_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> YoloFruitSliceBlueprint:
    return YoloFruitSliceBlueprint()
