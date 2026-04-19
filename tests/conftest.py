"""Pytest config + shared fixtures.

We put `cli/` on sys.path so test files can import the cli modules with
their bare names (`import coach`, `import content`, `import _log`, etc.) —
matching how chat.py imports them when run as a script.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DIR = REPO_ROOT / "cli"
BACKEND_DIR = REPO_ROOT / "web-py" / "backend"

for p in (CLI_DIR, BACKEND_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Force-reset the singleton so every test starts from a known logging state.
import _log  # noqa: E402

_log._CONFIGURED = False
for h in list(_log.logging.getLogger("voicecoach").handlers):
    _log.logging.getLogger("voicecoach").removeHandler(h)


def pytest_collection_modifyitems(config, items):
    """Mark tests in test_server.py as needing asyncio."""
    import pytest
    for item in items:
        if "test_server" in item.nodeid:
            item.add_marker(pytest.mark.asyncio)


# Sandbox session logs into a temp dir per pytest invocation so we never
# pollute ~/.voice-coach/sessions/ from test runs.
os.environ.setdefault(
    "VOICE_COACH_SESSION_DIR",
    str(REPO_ROOT / "tests" / ".tmp" / "sessions"),
)
