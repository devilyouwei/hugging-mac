"""Text-to-speech App blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.text_to_speech.config import TextToSpeechSettings
from hugging_mac_web.text_to_speech.manifest import TEXT_TO_SPEECH_MANIFEST
from hugging_mac_web.text_to_speech.routes import create_router


class TextToSpeechBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return TEXT_TO_SPEECH_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router(TextToSpeechSettings())


def create_blueprint() -> TextToSpeechBlueprint:
    return TextToSpeechBlueprint()
