"""Live transcription App blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.live_transcription.config import LiveTranscriptionSettings
from hugging_mac_web.live_transcription.manifest import LIVE_TRANSCRIPTION_MANIFEST
from hugging_mac_web.live_transcription.routes import create_router


class LiveTranscriptionBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return LIVE_TRANSCRIPTION_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router(LiveTranscriptionSettings())


def create_blueprint() -> LiveTranscriptionBlueprint:
    return LiveTranscriptionBlueprint()
