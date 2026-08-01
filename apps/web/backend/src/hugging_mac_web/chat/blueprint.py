"""Local chat App blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest
from hugging_mac_web.chat.config import ChatSettings
from hugging_mac_web.chat.manifest import CHAT_MANIFEST
from hugging_mac_web.chat.routes import create_router


class ChatBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return CHAT_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router(ChatSettings())


def create_blueprint() -> ChatBlueprint:
    return ChatBlueprint()
