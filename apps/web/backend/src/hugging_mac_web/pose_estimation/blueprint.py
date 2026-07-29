"""YOLOv8 Pose App blueprint."""

from __future__ import annotations

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.pose_estimation.config import PoseEstimationSettings
from hugging_mac_web.pose_estimation.manifest import POSE_ESTIMATION_MANIFEST
from hugging_mac_web.pose_estimation.routes import create_router


class PoseEstimationBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return POSE_ESTIMATION_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router(PoseEstimationSettings())


def create_blueprint() -> PoseEstimationBlueprint:
    return PoseEstimationBlueprint()
