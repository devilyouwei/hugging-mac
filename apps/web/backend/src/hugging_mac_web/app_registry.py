"""Explicit registry for business App blueprints."""

from __future__ import annotations

from enum import StrEnum
from threading import RLock

from hugging_mac_sdk.core.registry import ModelRegistry
from hugging_mac_sdk.errors import ResourceNotFoundError
from pydantic import BaseModel, ConfigDict, Field


class AppStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class AppModelRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    capabilities: tuple[str, ...]
    preferred_runtime: str | None = None
    required: bool = True


class AppManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    description: str
    tags: frozenset[str] = frozenset()
    version: str = "1"
    frontend_route: str
    api_prefix: str
    required_models: tuple[AppModelRequirement, ...] = ()


class AppSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: AppManifest
    status: AppStatus
    unavailable_reason: str | None = None


class AppRegistry:
    def __init__(self) -> None:
        self._apps: dict[str, AppSummary] = {}
        self._routes: set[str] = set()
        self._lock = RLock()

    def register(self, summary: AppSummary) -> None:
        manifest = summary.manifest
        with self._lock:
            if manifest.app_id in self._apps:
                raise ValueError(f"App is already registered: {manifest.app_id}")
            for route in (manifest.frontend_route, manifest.api_prefix):
                if route in self._routes:
                    raise ValueError(f"App route is already registered: {route}")
            self._apps[manifest.app_id] = summary
            self._routes.update((manifest.frontend_route, manifest.api_prefix))

    def register_manifest(
        self,
        manifest: AppManifest,
        models: ModelRegistry,
    ) -> AppSummary:
        reason = _validate_model_requirements(manifest, models)
        summary = AppSummary(
            manifest=manifest,
            status=AppStatus.UNAVAILABLE if reason else AppStatus.AVAILABLE,
            unavailable_reason=reason,
        )
        self.register(summary)
        return summary

    def list(self) -> tuple[AppSummary, ...]:
        with self._lock:
            return tuple(sorted(self._apps.values(), key=lambda item: item.manifest.app_id))

    def get(self, app_id: str) -> AppSummary | None:
        with self._lock:
            return self._apps.get(app_id)


def _validate_model_requirements(
    manifest: AppManifest,
    models: ModelRegistry,
) -> str | None:
    for requirement in manifest.required_models:
        try:
            model = models.get(requirement.model_id).manifest
        except ResourceNotFoundError:
            if requirement.required:
                return f"Required model is not registered: {requirement.model_id}"
            continue
        missing = set(requirement.capabilities) - model.capabilities
        if missing and requirement.required:
            return (
                f"Model {requirement.model_id} is missing capabilities: "
                f"{', '.join(sorted(missing))}"
            )
        if requirement.preferred_runtime is not None and not any(
            runtime.name == requirement.preferred_runtime for runtime in model.runtimes
        ):
            return (
                f"Model {requirement.model_id} does not declare runtime "
                f"{requirement.preferred_runtime}"
            )
    return None
