"""Command-line entry point for the local platform server."""

from __future__ import annotations

import uvicorn

from hugging_mac_web.config import WebSettings


def main() -> None:
    settings = WebSettings()
    uvicorn.run(
        "hugging_mac_web.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
