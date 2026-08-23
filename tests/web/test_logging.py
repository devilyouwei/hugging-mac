from __future__ import annotations

import logging
import re

import pytest
import structlog
from hugging_mac_web.config import WebSettings
from hugging_mac_web.shared.utils.log_util import configure_logging, get_logger


def test_terminal_logs_include_callsite_and_level_colors(capsys: object) -> None:
    configure_logging("DEBUG", colors=True)
    logger = get_logger("test")

    logger.debug("debug event")
    logger.info("info event")
    logger.warning("warning event")
    logger.error("error event")
    logger.critical("critical event")
    output = capsys.readouterr().err  # type: ignore[attr-defined]
    lines = output.splitlines()

    assert "\033[2;37m" in lines[0]
    assert "\033[36m" in lines[0]
    assert "\033[32m[INFO]" in lines[1]
    assert "\033[33m" in lines[2]
    assert "\033[31m" in lines[3]
    assert "\033[1;31m" in lines[4]
    assert "\033[36mtest\033[0m" in output
    assert "\033[35m[test_logging.py:" in output
    assert "\033[1;37minfo event\033[0m" in output
    assert "test_logging.py" in output
    assert "[test_logging.py:" in output
    assert "test_terminal_logs_include_callsite_and_level_colors]" in output
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}", output)
    assert "info event" in output
    assert "source=" not in output
    assert "[info     ]" not in output


def test_exception_log_contains_full_traceback(capsys: object) -> None:
    configure_logging("INFO", colors=False)
    logger = get_logger("test")

    try:
        raise ValueError("broken payload")
    except ValueError:
        logger.exception("request failed")

    output = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "request failed" in output
    assert "Traceback (most recent call last)" in output
    assert "[test_logging.py:" in output
    assert 'raise ValueError("broken payload")' in output
    assert "ValueError: broken payload" in output


def test_stdlib_logs_use_the_same_detailed_renderer(capsys: object) -> None:
    configure_logging("INFO", colors=False)

    logging.getLogger("sdk.worker").warning("queue pressure", extra={"queue_size": 7})

    output = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "queue pressure" in output
    assert "[WARNING] sdk.worker" in output
    assert "queue_size=7" in output
    assert "[WARNING]" in output
    assert "test_logging.py" in output


def test_logger_created_before_configuration_uses_final_renderer(capsys: object) -> None:
    structlog.reset_defaults()
    logger = get_logger("early.component")

    configure_logging("INFO", colors=False)
    logger.info("configured later")

    output = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "[INFO] early.component [test_logging.py:" in output
    assert "configured later" in output
    assert "source=" not in output
    assert "[info     ]" not in output


def test_uvicorn_access_logs_omit_internal_callsite(capsys: object) -> None:
    configure_logging("INFO", colors=False)

    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:50000",
        "POST",
        "/api/v1/apps/live-transcription/models/load",
        "1.1",
        200,
        extra={"color_message": "ignored"},
    )

    output = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "[INFO] uvicorn.access" in output
    assert "POST /api/v1/apps/live-transcription/models/load" in output
    assert "source=" not in output
    assert "color_message=" not in output


def test_web_log_level_defaults_to_info_and_accepts_debug_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEB_LOG_LEVEL", raising=False)
    assert WebSettings(_env_file=None).log_level == "INFO"

    monkeypatch.setenv("WEB_LOG_LEVEL", "debug")
    assert WebSettings(_env_file=None).log_level == "DEBUG"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        configure_logging("verbose")
