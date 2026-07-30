"""Yolo Pose Follow blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.yolo_pose_follow.manifest import YOLO_POSE_FOLLOW_MANIFEST
from hugging_mac_web.yolo_pose_follow.routes import create_router


class YoloPoseFollowBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return YOLO_POSE_FOLLOW_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> YoloPoseFollowBlueprint:
    return YoloPoseFollowBlueprint()
