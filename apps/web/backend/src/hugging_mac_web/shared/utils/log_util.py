"""Consistent structured logging for application and third-party loggers."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from typing import cast

import structlog
from structlog.processors import CallsiteParameter
from structlog.typing import EventDict, Processor

_RESET = "\033[0m"
_TIMESTAMP_COLOR = "\033[2;37m"
_LOGGER_COLOR = "\033[36m"
_LOCATION_COLOR = "\033[35m"
_EVENT_COLOR = "\033[1;37m"
_LEVEL_COLORS = {
    "debug": "\033[36m",
    "info": "\033[32m",
    "warning": "\033[33m",
    "warn": "\033[33m",
    "error": "\033[31m",
    "exception": "\033[31m",
    "critical": "\033[1;31m",
}
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _add_millisecond_timestamp(
    _logger: object,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Use local wall time with timezone and stable millisecond precision."""

    event_dict["timestamp"] = (
        datetime.now().astimezone().isoformat(sep=" ", timespec="milliseconds")
    )
    return event_dict


def _add_execution_context(
    _logger: object,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add the active asyncio task when logging from asynchronous code."""

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        event_dict["task_name"] = task.get_name()
    return event_dict


class CompactTerminalRenderer:
    """Render compact single-line records without padded columns.

    INFO deliberately has no style so it uses the terminal's current default.
    Tracebacks remain multiline and are appended only to exception records.
    """

    def __init__(self, *, colors: bool) -> None:
        self._colors = colors

    def __call__(
        self,
        logger: object,
        method_name: str,
        event_dict: EventDict,
    ) -> str:
        fields = dict(event_dict)
        level = str(fields.pop("level", method_name)).upper()
        timestamp = str(fields.pop("timestamp", ""))
        event = str(fields.pop("event", ""))
        logger_name = str(fields.pop("logger", fields.get("component", "application")))
        fields.pop("component", None)
        fields.pop("color_message", None)

        filename = fields.pop("filename", None)
        lineno = fields.pop("lineno", None)
        func_name = fields.pop("func_name", None)
        task_name = fields.pop("task_name", None)
        exception = fields.pop("exception", None)
        stack = fields.pop("stack", None)

        location = ""
        # Framework lifecycle/access records are already self-describing. Their
        # internal Uvicorn source location only adds noise. The logger name is
        # already the module identity, so application records only need file:line.
        if not logger_name.startswith("uvicorn.") and filename is not None:
            source = f"{filename}:{lineno}" if lineno is not None else str(filename)
            location = f"[{source}{f' {func_name}' if func_name else ''}]"

        timestamp_token = self._style(timestamp, _TIMESTAMP_COLOR)
        level_token = self._style(f"[{level}]", _LEVEL_COLORS.get(level.lower()))
        logger_token = self._style(logger_name, _LOGGER_COLOR)
        location_token = self._style(location, _LOCATION_COLOR) if location else ""
        event_token = self._style(event, _EVENT_COLOR)
        rendered = (
            f"{timestamp_token} {level_token} {logger_token}"
            f"{f' {location_token}' if location_token else ''} {event_token}"
        )
        if task_name is not None:
            rendered += f" task={_render_value(task_name)}"
        if fields:
            rendered += " " + " ".join(
                f"{key}={_render_value(value)}" for key, value in sorted(fields.items())
            )
        if stack:
            rendered += f"\n{stack}"
        if exception:
            rendered += f"\n{exception}"

        return rendered

    def _style(self, value: str, color: str | None) -> str:
        return f"{color}{value}{_RESET}" if self._colors and color else value


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return repr(value) if any(character.isspace() for character in value) else value
    return repr(value)


def configure_logging(
    level: str = "INFO",
    *,
    colors: bool | None = None,
) -> None:
    """Configure one renderer for structlog, stdlib, Uvicorn, and SDK logs.

    Exception calls include their complete traceback. Normal records include local
    zoned time, level, logger/component, compact callsite, and asyncio task context.
    ``colors=None`` enables colors only for an interactive stderr terminal.
    """

    normalized_level = level.upper()
    if normalized_level not in _LOG_LEVELS:
        supported = ", ".join(_LOG_LEVELS)
        raise ValueError(f"Unsupported log level {level!r}; expected one of: {supported}")
    resolved_level = _LOG_LEVELS[normalized_level]
    use_colors = sys.stderr.isatty() if colors is None else colors
    callsite = structlog.processors.CallsiteParameterAdder(
        parameters={
            CallsiteParameter.FILENAME,
            CallsiteParameter.LINENO,
            CallsiteParameter.FUNC_NAME,
        }
    )
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _add_millisecond_timestamp,
        _add_execution_context,
        callsite,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            CompactTerminalRenderer(colors=use_colors),
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Uvicorn installs dedicated handlers by default. Route them through the
    # application formatter so access/server logs are consistent and not duplicated.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(resolved_level)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a lazy named logger that follows the latest global configuration.

    Application modules create loggers while they are imported, before the server
    configures logging. Calling ``bind()`` here would eagerly freeze structlog's
    fallback renderer and produce a second, inconsistent log format.
    """

    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger(component, component=component),
    )
