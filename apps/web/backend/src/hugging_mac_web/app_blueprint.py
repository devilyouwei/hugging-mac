"""Contract implemented by independently packaged business applications."""

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter

from hugging_mac_web.app_registry import AppManifest


class AppBlueprint(Protocol):
    """A side-effect-free App registration entry point."""

    @property
    def manifest(self) -> AppManifest: ...

    def create_router(self) -> APIRouter: ...
