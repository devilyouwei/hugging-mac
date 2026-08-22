"""Command-line entry point for the local platform server."""

from __future__ import annotations

import uvicorn

from hugging_mac_web.config import WebSettings
from hugging_mac_web.shared.utils.log_util import configure_logging


def main() -> None:
    settings = WebSettings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "hugging_mac_web.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
    )
