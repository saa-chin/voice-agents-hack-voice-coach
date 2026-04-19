"""Coverage for cli/_exit.py — exit codes + die() helper."""
from __future__ import annotations

import logging

import _exit
import _log
import pytest
from _exit import ExitCode, die


@pytest.fixture(autouse=True)
def _reset_logger():
    """Force the logger to re-attach its StreamHandler to the test's
    redirected stderr each test, so capsys can see emitted records."""
    parent = logging.getLogger("voicecoach")
    for h in list(parent.handlers):
        parent.removeHandler(h)
    _log._CONFIGURED = False
    yield
    for h in list(parent.handlers):
        parent.removeHandler(h)
    _log._CONFIGURED = False


def test_all_exit_codes_are_unique():
    seen: dict[int, str] = {}
    for name in dir(ExitCode):
        if name.startswith("_"):
            continue
        val = getattr(ExitCode, name)
        if not isinstance(val, int):
            continue
        assert val not in seen, f"{name} duplicates {seen[val]} ({val})"
        seen[val] = name


def test_exit_code_ranges():
    """Codes follow the documented ranges."""
    assert ExitCode.OK == 0
    assert ExitCode.USER_ABORT == 130
    # env: 10-19
    assert 10 <= ExitCode.ENV_MISSING_TOOL <= 19
    assert 10 <= ExitCode.ENV_MISSING_LIB <= 19
    assert 10 <= ExitCode.ENV_MISSING_BINDINGS <= 19
    assert 10 <= ExitCode.ENV_MISSING_PYTHON_DEP <= 19
    assert 10 <= ExitCode.ENV_NO_AUDIO_DEVICE <= 19
    # setup: 20-29
    assert 20 <= ExitCode.SETUP_DOWNLOAD_FAILED <= 29
    assert 20 <= ExitCode.SETUP_MODEL_LOAD_FAILED <= 29
    assert 20 <= ExitCode.SETUP_LIB_LINK_FAILED <= 29
    # runtime: 30-39
    assert 30 <= ExitCode.RUNTIME_AUDIO_FAILED <= 39
    assert 30 <= ExitCode.RUNTIME_MODEL_FAILED <= 39
    assert 30 <= ExitCode.RUNTIME_BAD_JSON <= 39
    # config: 40-49
    assert 40 <= ExitCode.CONFIG_INVALID <= 49


def test_die_exits_with_given_code():
    with pytest.raises(SystemExit) as exc:
        die(ExitCode.ENV_MISSING_TOOL, "missing widget")
    assert exc.value.code == ExitCode.ENV_MISSING_TOOL


def test_die_exits_with_arbitrary_code():
    with pytest.raises(SystemExit) as exc:
        die(42, "custom failure")
    assert exc.value.code == 42


def test_die_logs_message_at_error_level(capsys):
    """Our `voicecoach` logger sets propagate=False, so caplog (rooted at
    the standard root logger) misses the records. Read stderr directly."""
    with pytest.raises(SystemExit):
        die(ExitCode.RUNTIME_BAD_JSON, "bad parse", "hint A", "hint B")
    text = capsys.readouterr().err
    assert "bad parse" in text
    assert f"exit {ExitCode.RUNTIME_BAD_JSON}" in text
    assert "hint A" in text
    assert "hint B" in text
    # Hints rendered as remediation arrows.
    assert "->" in text
    # Emitted at ERROR level.
    assert "ERROR" in text


def test_die_with_no_hints():
    """Calling die() without hints should still work."""
    with pytest.raises(SystemExit):
        die(ExitCode.CONFIG_INVALID, "just the message")


def test_die_typing_noreturn():
    """die() is annotated as NoReturn — verify it really doesn't return."""
    import typing
    hints = typing.get_type_hints(die)
    assert hints["return"] is typing.NoReturn
