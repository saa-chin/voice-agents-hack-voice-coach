"""Exit codes and a `die()` helper for the voice-coach CLI.

A small, stable set so wrappers (run-cli, CI, demo scripts) can branch on
*why* something failed without parsing log text.

Numeric ranges:
    0       success
    1-9     reserved (argparse / shell)
    10-19   environment problems (missing tools, dylib, bindings, deps, no audio device)
    20-29   setup problems (model download, model load, lib symlink)
    30-39   runtime problems (audio capture, model inference, malformed JSON)
    40-49   config problems (invalid args, bad content)
    130     SIGINT (POSIX convention)
"""
from __future__ import annotations

import sys
from typing import NoReturn

import _log


class ExitCode:
    OK                       = 0
    USER_ABORT               = 130

    ENV_MISSING_TOOL         = 10
    ENV_MISSING_LIB          = 11
    ENV_MISSING_BINDINGS     = 12
    ENV_MISSING_PYTHON_DEP   = 13
    ENV_NO_AUDIO_DEVICE      = 14

    SETUP_DOWNLOAD_FAILED    = 20
    SETUP_MODEL_LOAD_FAILED  = 21
    SETUP_LIB_LINK_FAILED    = 22

    RUNTIME_AUDIO_FAILED     = 30
    RUNTIME_MODEL_FAILED     = 31
    RUNTIME_BAD_JSON         = 32

    CONFIG_INVALID           = 40


def die(code: int, msg: str, *hints: str) -> NoReturn:
    """Log an ERROR with optional remediation hints, then exit with `code`."""
    log = _log.get("exit")
    log.error("%s (exit %d)", msg, code)
    for hint in hints:
        log.error("  -> %s", hint)
    sys.exit(code)
