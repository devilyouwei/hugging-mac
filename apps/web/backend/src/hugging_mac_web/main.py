"""FastAPI application factory."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hugging_mac_web.app_blueprint import AppBlueprint
from hugging_mac_web.config import WebSettings
from hugging_mac_web.context import create_context
from hugging_mac_web.error_handlers import install_error_handlers
from hugging_mac_web.index import create_index_router
from hugging_mac_web.instance_segmentation import (
    create_blueprint as create_segmentation_blueprint,
)
from hugging_mac_web.live_transcription import (
    create_blueprint as create_live_transcription_blueprint,
)
from hugging_mac_web.media import create_media_router
from hugging_mac_web.middleware import TraceIdMiddleware
from hugging_mac_web.models import create_models_router
from hugging_mac_web.object_detection import create_blueprint as create_detection_blueprint
from hugging_mac_web.pose_estimation import create_blueprint as create_pose_blueprint
from hugging_mac_web.shared.utils.log_util import configure_logging
from hugging_mac_web.system import create_system_router
from hugging_mac_web.text_to_speech import (
    create_blueprint as create_text_to_speech_blueprint,
)
from hugging_mac_web.yolo_pose_follow import create_blueprint as create_pose_follow_blueprint


def create_app(
    settings: WebSettings | None = None,
    *,
    blueprints: Iterable[AppBlueprint] | None = None,
) -> FastAPI:
    resolved = settings or WebSettings()
    selected_blueprints = (
        tuple(blueprints)
        if blueprints is not None
        else (
            create_detection_blueprint(),
            create_pose_blueprint(),
            create_segmentation_blueprint(),
            create_live_transcription_blueprint(),
            create_text_to_speech_blueprint(),
            create_pose_follow_blueprint(),
        )
    )
    registered_blueprints = tuple(
        (blueprint, blueprint.create_router()) for blueprint in selected_blueprints
    )
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        context = create_context(resolved)
        app.state.context = context
        try:
            for blueprint, _ in registered_blueprints:
                context.apps.register_manifest(blueprint.manifest, context.models.registry)
            yield
        finally:
            with contextlib.suppress(Exception):
                await context.models.instances.unload_all(force=True)
            context.close()

    app = FastAPI(
        title="hugging-mac",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.parsed_cors_origins),
        allow_credentials="*" not in resolved.parsed_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-trace-id"],
    )
    install_error_handlers(app)
    app.include_router(create_index_router())
    app.include_router(create_models_router())
    app.include_router(create_media_router())
    app.include_router(create_system_router())
    for _, router in registered_blueprints:
        app.include_router(router)

    @app.get("/", tags=["platform"])
    async def root() -> dict[str, str]:
        return {
            "name": "hugging-mac",
            "status": "backend-ready",
            "docs": "/docs",
        }

    return app


app = create_app()
