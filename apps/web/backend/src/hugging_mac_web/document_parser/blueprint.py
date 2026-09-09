"""Document Parser app blueprint."""

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest

from .manifest import DOCUMENT_PARSER_MANIFEST
from .routes import create_router


class DocumentParserBlueprint:
    @property
    def manifest(self) -> AppManifest:
        return DOCUMENT_PARSER_MANIFEST

    def create_router(self) -> APIRouter:
        return create_router()


def create_blueprint() -> DocumentParserBlueprint:
    return DocumentParserBlueprint()
