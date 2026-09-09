"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hugging_mac_web.app_blueprint import AppBlueprint
from hugging_mac_web.chat import create_blueprint as create_chat_blueprint
from hugging_mac_web.config import WebSettings
from hugging_mac_web.document_parser import create_blueprint as create_document_parser_blueprint
from hugging_mac_web.error_handlers import install_error_handlers
from hugging_mac_web.index import create_index_router
from hugging_mac_web.instance_segmentation import (
    create_blueprint as create_segmentation_blueprint,
)
from hugging_mac_web.lifecycle import create_lifespan
from hugging_mac_web.live_transcription import (
    create_blueprint as create_live_transcription_blueprint,
)
from hugging_mac_web.media import create_media_router
from hugging_mac_web.middleware import TraceIdMiddleware
from hugging_mac_web.models import create_models_router
from hugging_mac_web.object_detection import create_blueprint as create_detection_blueprint
from hugging_mac_web.palm_thunder import create_blueprint as create_palm_thunder_blueprint
from hugging_mac_web.palm_trace import create_blueprint as create_palm_trace_blueprint
from hugging_mac_web.pose_estimation import create_blueprint as create_pose_blueprint
from hugging_mac_web.system import create_system_router
from hugging_mac_web.text_to_speech import (
    create_blueprint as create_text_to_speech_blueprint,
)
from hugging_mac_web.yolo_fruit_slice import create_blueprint as create_fruit_slice_blueprint
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
            create_chat_blueprint(),
            create_document_parser_blueprint(),
            create_pose_follow_blueprint(),
            create_fruit_slice_blueprint(),
            create_palm_thunder_blueprint(),
            create_palm_trace_blueprint(),
        )
    )
    registered_blueprints = tuple(
        (blueprint, blueprint.create_router()) for blueprint in selected_blueprints
    )
    app = FastAPI(
        title="hugging-mac",
        version="0.1.0",
        lifespan=create_lifespan(resolved, selected_blueprints),
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
