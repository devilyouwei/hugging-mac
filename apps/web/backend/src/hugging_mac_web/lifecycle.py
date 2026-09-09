"""Single ownership boundary for backend startup and shutdown."""

from __future__ import annotations

import asyncio
import faulthandler
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from hugging_mac_web.app_blueprint import AppBlueprint
from hugging_mac_web.config import WebSettings
from hugging_mac_web.context import PlatformContext, create_context
from hugging_mac_web.shared.utils.log_util import configure_logging, get_logger

logger = get_logger("platform.lifecycle")


def create_lifespan(
    settings: WebSettings,
    blueprints: Iterable[AppBlueprint],
) -> Callable[[FastAPI], Any]:
    """Build FastAPI's lifespan with every platform service managed here."""

    registered = tuple(blueprints)
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Development servers may replace logging after module import, so the
        # lifecycle boundary intentionally reapplies the shared configuration.
        configure_logging(settings.log_level)
        _install_process_diagnostics()
        context = create_context(settings)
        app.state.context = context
        logger.info("platform_startup_started")
        try:
            for blueprint in registered:
                context.apps.register_manifest(blueprint.manifest, context.models.registry)
            await context.document_parser.start()
            logger.info(
                "platform_startup_completed",
                app_count=len(registered),
            )
            yield
        finally:
            logger.info("platform_shutdown_started")
            await close_context(context)
            logger.info("platform_shutdown_completed")

    return lifespan


async def close_context(context: PlatformContext) -> None:
    """Stop services in dependency order while still attempting every cleanup."""

    await _shutdown_step("document_parser", context.document_parser.close)
    await _shutdown_step("downloads", context.downloads.close)
    await _shutdown_step(
        "model_instances",
        lambda: context.models.instances.unload_all(force=True),
    )
    await _shutdown_step("document_store", context.documents.close)


async def _shutdown_step(name: str, operation: Callable[[], Any]) -> None:
    try:
        result = operation()
        if hasattr(result, "__await__"):
            await result
    except Exception as error:
        logger.exception(
            "platform_shutdown_step_failed",
            service=name,
            error_type=type(error).__name__,
        )


def _install_process_diagnostics() -> None:
    """Expose Python exceptions and native fatal signals in the server log."""

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)

    loop = asyncio.get_running_loop()

    def report_asyncio_error(
        _loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        exception = context.get("exception")
        logger.error(
            "unhandled_asyncio_error",
            message=context.get("message", "no message"),
            exc_info=(type(exception), exception, exception.__traceback__)
            if isinstance(exception, BaseException)
            else None,
        )

    loop.set_exception_handler(report_asyncio_error)
