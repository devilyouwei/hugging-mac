"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import cast

import structlog
from structlog.processors import CallsiteParameter

_RESET = "\033[0m"
_LEVEL_COLORS = {
    "debug": "\033[36m",
    "warning": "\033[33m",
    "warn": "\033[33m",
    "error": "\033[31m",
    "exception": "\033[31m",
    "critical": "\033[1;31m",
}


class TerminalLogRenderer:
    """Render readable terminal logs and color the complete line by level.

    INFO deliberately has no style so it uses the terminal's current default.
    """

    def __init__(self, *, colors: bool) -> None:
        self._colors = colors
        self._renderer = structlog.dev.ConsoleRenderer(
            colors=False,
            exception_formatter=structlog.dev.plain_traceback,
            sort_keys=False,
        )

    def __call__(
        self,
        logger: object,
        method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> str:
        level = str(event_dict.get("level", method_name)).lower()
        rendered = self._renderer(logger, method_name, event_dict)
        color = _LEVEL_COLORS.get(level) if self._colors else None
        return f"{color}{rendered}{_RESET}" if color else rendered


def configure_logging(
    level: str = "INFO",
    *,
    colors: bool | None = None,
) -> None:
    """Configure business logs for a local terminal.

    ``colors=None`` enables colors only when stderr is attached to a TTY.
    Tests and embedding applications can force either behavior explicitly.
    """

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    use_colors = sys.stderr.isatty() if colors is None else colors
    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder(
                parameters={
                    CallsiteParameter.PATHNAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.FUNC_NAME,
                }
            ),
            structlog.processors.StackInfoRenderer(),
            TerminalLogRenderer(colors=use_colors),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger(component=component),
    )
