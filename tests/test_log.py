"""Coverage for cli/_log.py — leveled logger + color formatter."""
from __future__ import annotations

import io
import logging
import sys

import _log
import pytest


@pytest.fixture(autouse=True)
def reset_log_singleton():
    """Each test starts from a fresh _log state."""
    _log._CONFIGURED = False
    parent = logging.getLogger("voicecoach")
    for h in list(parent.handlers):
        parent.removeHandler(h)
    parent.setLevel(logging.NOTSET)
    yield
    _log._CONFIGURED = False
    for h in list(parent.handlers):
        parent.removeHandler(h)
    parent.setLevel(logging.NOTSET)


def test_get_returns_logger_under_namespace():
    log = _log.get("foo")
    assert log.name == "voicecoach.foo"
    assert isinstance(log, logging.Logger)


def test_get_does_not_double_prefix():
    log = _log.get("voicecoach.bar")
    assert log.name == "voicecoach.bar"


def test_configure_default_level_is_info(monkeypatch):
    monkeypatch.delenv("VOICE_COACH_LOG_LEVEL", raising=False)
    _log.configure()
    assert logging.getLogger("voicecoach").level == logging.INFO


def test_configure_explicit_string_level():
    _log.configure("DEBUG")
    assert logging.getLogger("voicecoach").level == logging.DEBUG


def test_configure_explicit_int_level():
    _log.configure(logging.WARNING)
    assert logging.getLogger("voicecoach").level == logging.WARNING


def test_configure_lowercase_level():
    _log.configure("warning")
    assert logging.getLogger("voicecoach").level == logging.WARNING


def test_configure_unknown_level_falls_back_to_info():
    _log.configure("NOPE")
    assert logging.getLogger("voicecoach").level == logging.INFO


def test_configure_env_var_override(monkeypatch):
    monkeypatch.setenv("VOICE_COACH_LOG_LEVEL", "ERROR")
    _log.configure()
    assert logging.getLogger("voicecoach").level == logging.ERROR


def test_configure_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("VOICE_COACH_LOG_LEVEL", "ERROR")
    _log.configure("DEBUG")
    assert logging.getLogger("voicecoach").level == logging.DEBUG


def test_configure_attaches_handler_only_once():
    _log.configure()
    _log.configure()
    _log.configure()
    parent = logging.getLogger("voicecoach")
    assert len(parent.handlers) == 1


def test_configure_second_call_changes_level_only():
    _log.configure("DEBUG")
    _log.configure("ERROR")
    parent = logging.getLogger("voicecoach")
    assert len(parent.handlers) == 1
    assert parent.level == logging.ERROR


def test_get_auto_configures_if_not_yet_configured():
    assert not _log._CONFIGURED
    log = _log.get("auto")
    log.info("just to ensure no exception")
    assert _log._CONFIGURED


def test_parent_logger_does_not_propagate():
    """We don't want voicecoach.* messages bubbling into the root logger."""
    _log.configure()
    assert logging.getLogger("voicecoach").propagate is False


def test_color_formatter_no_color_for_pipe(monkeypatch):
    """When stderr is not a TTY, the formatter must not emit ANSI codes."""
    fmt = _log._ColorFormatter(use_color=False)
    rec = logging.LogRecord(
        "voicecoach.x", logging.INFO, "f", 1, "hello", None, None,
    )
    out = fmt.format(rec)
    assert "\033[" not in out
    assert "hello" in out
    assert "INFO" in out


def test_color_formatter_with_color_emits_ansi():
    fmt = _log._ColorFormatter(use_color=True)
    rec = logging.LogRecord(
        "voicecoach.x", logging.WARNING, "f", 1, "warned", None, None,
    )
    out = fmt.format(rec)
    assert "\033[" in out
    assert "warned" in out
    assert _log._RESET in out


def test_color_formatter_unknown_level_no_color():
    """Custom levels without an entry in the color map fall through cleanly."""
    fmt = _log._ColorFormatter(use_color=True)
    rec = logging.LogRecord(
        "voicecoach.x", 5, "f", 1, "low", None, None,
    )
    rec.levelname = "TRACE"  # not in _LEVEL_COLORS
    out = fmt.format(rec)
    # No color codes for unknown levels.
    assert "\033[" not in out
    assert "low" in out


def test_logging_actually_emits_to_stderr(capsys):
    _log.configure("INFO")
    log = _log.get("sink")
    log.info("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.err
    assert "voicecoach.sink" in captured.err
