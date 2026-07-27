from __future__ import annotations

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

    assert "\033[36m" in lines[0]
    assert "\033[" not in lines[1]
    assert "\033[33m" in lines[2]
    assert "\033[31m" in lines[3]
    assert "\033[1;31m" in lines[4]
    assert "test_logging.py" in output
    assert "lineno=" in output
    assert "func_name=test_terminal_logs_include_callsite_and_level_colors" in output


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
    assert 'raise ValueError("broken payload")' in output
    assert "ValueError: broken payload" in output
