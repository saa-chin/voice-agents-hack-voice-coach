"""Logging setup for the voice-coach CLI.

Use `_log.get(name)` to obtain a logger. The first call configures the
"voicecoach" parent logger; subsequent calls just return child loggers.

Level resolution order (first wins):
    1. explicit `configure(level=...)` argument
    2. env var VOICE_COACH_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)
    3. INFO

Output goes to stderr so it never interleaves with the conversational UI
on stdout. ANSI colors are emitted only when stderr is a TTY.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_PARENT = "voicecoach"

_LEVEL_COLORS = {
    "DEBUG":    "\033[2m",
    "INFO":     "\033[36m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self._use_color:
            return msg
        color = _LEVEL_COLORS.get(record.levelname, "")
        return f"{color}{msg}{_RESET}" if color else msg


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get("VOICE_COACH_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    resolved = getattr(logging, str(level).upper(), None)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure(level: str | int | None = None) -> None:
    """Configure the parent logger. Safe to call multiple times."""
    global _CONFIGURED
    parent = logging.getLogger(_PARENT)
    parent.setLevel(_coerce_level(level))
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_ColorFormatter(use_color=sys.stderr.isatty()))
    parent.addHandler(handler)
    parent.propagate = False
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    """Return a child logger under the `voicecoach` namespace."""
    if not _CONFIGURED:
        configure()
    if not name.startswith(_PARENT):
        name = f"{_PARENT}.{name}"
    return logging.getLogger(name)
