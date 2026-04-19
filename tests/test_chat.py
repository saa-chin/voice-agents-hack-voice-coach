"""Coverage for cli/chat.py — pure helpers + env/setup paths via monkeypatch.

Excluded from these tests (need real Cactus + microphone):
  - text_chat()'s REPL inner loop (input() interactive)
  - voice_chat() (mic + Cactus)
  - coach_chat() (delegates to coach.coach_mode)
  - _cactus_complete_audio (FFI; covered indirectly by test_server.py with mock)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import chat
import pytest
from _exit import ExitCode


# ---- _split_sentences (pure) --------------------------------------------

class TestSplitSentences:
    def test_no_sentences_yet(self):
        out, leftover = chat._split_sentences("hello world")
        assert out == []
        assert leftover == "hello world"

    def test_single_complete_sentence(self):
        out, leftover = chat._split_sentences("Hello world. ")
        assert out == ["Hello world."]
        assert leftover == ""

    def test_multiple_sentences(self):
        out, leftover = chat._split_sentences("First. Second! Third? ")
        assert out == ["First.", "Second!", "Third?"]
        assert leftover == ""

    def test_partial_trailing_sentence(self):
        out, leftover = chat._split_sentences("Done. Mid")
        assert out == ["Done."]
        assert leftover == "Mid"

    def test_handles_double_punctuation(self):
        """e.g. ellipsis-style endings, !!! etc."""
        out, leftover = chat._split_sentences("Wow!!! ")
        assert out == ["Wow!!!"]
        assert leftover == ""

    def test_handles_quotes_after_punctuation(self):
        out, leftover = chat._split_sentences('"Done." ')
        # The closing quote is grouped with the period.
        assert out == ['"Done."']
        assert leftover == ""

    def test_decimal_points_dont_split(self):
        """3.14 should not be treated as 'three' . 'fourteen'.
        The splitter requires a space/EOL after punctuation to count it as a
        sentence boundary, so internal decimals are kept intact."""
        out, leftover = chat._split_sentences("Pi is 3.14 not 3.15.")
        # Whole thing is one sentence (final '.' at EOL = boundary).
        assert out == ["Pi is 3.14 not 3.15."]
        assert leftover == ""

    def test_decimal_followed_by_sentence_end(self):
        out, leftover = chat._split_sentences("Pi is 3.14. Done. ")
        assert out == ["Pi is 3.14.", "Done."]
        assert leftover == ""

    def test_decimal_no_trailing_punct_is_leftover(self):
        out, leftover = chat._split_sentences("Pi is 3.14 not 3.15")
        # No trailing punctuation → entire thing buffered as leftover.
        assert out == []
        assert leftover == "Pi is 3.14 not 3.15"

    def test_empty_string(self):
        out, leftover = chat._split_sentences("")
        assert out == []
        assert leftover == ""


# ---- ensure_lib_discoverable -------------------------------------------

class TestEnsureLibDiscoverable:
    def test_no_op_when_symlink_already_exists(self, monkeypatch, tmp_path):
        existing = tmp_path / "libcactus.dylib"
        existing.write_text("dummy")
        monkeypatch.setattr(chat, "EXPECTED_LIB", existing)
        # Should not touch the filesystem further.
        chat.ensure_lib_discoverable()  # no exception

    def test_dies_when_brew_dylib_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "EXPECTED_LIB", tmp_path / "missing.dylib")
        monkeypatch.setattr(chat, "CACTUS_LIB", tmp_path / "also-missing.dylib")
        with pytest.raises(SystemExit) as exc:
            chat.ensure_lib_discoverable()
        assert exc.value.code == ExitCode.ENV_MISSING_LIB

    def test_creates_symlink_when_brew_present(self, monkeypatch, tmp_path):
        brew = tmp_path / "brew" / "libcactus.dylib"
        brew.parent.mkdir(parents=True)
        brew.write_text("real lib bytes")
        target = tmp_path / "deep" / "nested" / "libcactus.dylib"
        monkeypatch.setattr(chat, "EXPECTED_LIB", target)
        monkeypatch.setattr(chat, "CACTUS_LIB", brew)
        chat.ensure_lib_discoverable()
        assert target.is_symlink()
        assert target.resolve() == brew.resolve()

    def test_dies_with_lib_link_failed_on_oserror(self, monkeypatch, tmp_path):
        brew = tmp_path / "libcactus.dylib"
        brew.write_text("dummy")

        class FakePath:
            def __init__(self, p): self._p = p
            def exists(self): return False
            def is_symlink(self): return False
            @property
            def parent(self):
                class P:
                    def mkdir(self, **kw): raise OSError("readonly fs")
                return P()
            def symlink_to(self, *a, **kw): raise OSError("nope")
            def __str__(self): return str(self._p)

        fake_target = FakePath(tmp_path / "x.dylib")
        monkeypatch.setattr(chat, "EXPECTED_LIB", fake_target)
        monkeypatch.setattr(chat, "CACTUS_LIB", brew)
        with pytest.raises(SystemExit) as exc:
            chat.ensure_lib_discoverable()
        assert exc.value.code == ExitCode.SETUP_LIB_LINK_FAILED


# ---- ensure_model -------------------------------------------------------

class TestEnsureModel:
    def test_returns_existing_path_without_download(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "CACTUS_WEIGHTS_DIR", tmp_path)
        weights = tmp_path / "gemma-4-e2b-it"
        weights.mkdir()
        (weights / "weights.bin").write_text("x")
        # subprocess.run must NOT be called.
        monkeypatch.setattr(
            chat.subprocess, "run",
            lambda *a, **kw: pytest.fail("subprocess.run should not run when weights present"),
        )
        out = chat.ensure_model("google/gemma-4-e2b-it")
        assert out == weights

    def test_runs_download_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "CACTUS_WEIGHTS_DIR", tmp_path)
        calls: list[list[str]] = []

        def fake_run(args, check):
            calls.append(args)
            # Simulate cactus dropping a file in the weights dir.
            (tmp_path / "gemma-4-e2b-it").mkdir()
            (tmp_path / "gemma-4-e2b-it" / "x.bin").write_text("y")
            return subprocess.CompletedProcess(args=args, returncode=0)

        monkeypatch.setattr(chat.subprocess, "run", fake_run)
        out = chat.ensure_model("google/gemma-4-e2b-it")
        assert out == tmp_path / "gemma-4-e2b-it"
        assert calls == [["cactus", "download", "google/gemma-4-e2b-it"]]

    def test_dies_if_cactus_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "CACTUS_WEIGHTS_DIR", tmp_path)

        def fake_run(args, check):
            raise FileNotFoundError(f"no such file: {args[0]}")

        monkeypatch.setattr(chat.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            chat.ensure_model("google/gemma-4-e2b-it")
        assert exc.value.code == ExitCode.ENV_MISSING_TOOL

    def test_dies_if_download_returns_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "CACTUS_WEIGHTS_DIR", tmp_path)

        def fake_run(args, check):
            raise subprocess.CalledProcessError(1, args)

        monkeypatch.setattr(chat.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            chat.ensure_model("google/gemma-4-e2b-it")
        assert exc.value.code == ExitCode.SETUP_DOWNLOAD_FAILED

    def test_dies_if_download_succeeds_but_weights_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(chat, "CACTUS_WEIGHTS_DIR", tmp_path)

        def fake_run(args, check):
            # Pretend success but never create the weights dir.
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(chat.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            chat.ensure_model("google/gemma-4-e2b-it")
        assert exc.value.code == ExitCode.SETUP_DOWNLOAD_FAILED


# ---- speak_blocking ----------------------------------------------------

class FakeProc:
    def __init__(self, *a, **kw):
        self.args = a[0] if a else kw.get("args")
        self.terminated = False
        self._poll = None
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        return 0

    def terminate(self):
        self.terminated = True
        self._poll = -15

    def poll(self):
        return self._poll


class TestSpeakBlocking:
    def setup_method(self):
        # Reset singleton state.
        chat._SAY_PROCESS = None

    def test_noop_on_empty_text(self, monkeypatch):
        called = []
        monkeypatch.setattr(chat.subprocess, "Popen", lambda *a, **kw: called.append(a))
        monkeypatch.setattr(chat.shutil, "which", lambda c: "/usr/bin/say")
        chat.speak_blocking("", None)
        chat.speak_blocking("   ", None)
        assert called == []

    def test_noop_when_say_unavailable(self, monkeypatch):
        called = []
        monkeypatch.setattr(chat.shutil, "which", lambda c: None)
        monkeypatch.setattr(chat.subprocess, "Popen", lambda *a, **kw: called.append(a))
        chat.speak_blocking("hello", None)
        assert called == []

    def test_invokes_say_with_text(self, monkeypatch):
        invocations: list[list[str]] = []

        def fake_popen(args):
            invocations.append(args)
            return FakeProc(args)

        monkeypatch.setattr(chat.shutil, "which", lambda c: "/usr/bin/say")
        monkeypatch.setattr(chat.subprocess, "Popen", fake_popen)
        chat.speak_blocking("hello world", None)
        assert invocations == [["say", "--", "hello world"]]

    def test_invokes_say_with_voice(self, monkeypatch):
        invocations: list[list[str]] = []

        def fake_popen(args):
            invocations.append(args)
            return FakeProc(args)

        monkeypatch.setattr(chat.shutil, "which", lambda c: "/usr/bin/say")
        monkeypatch.setattr(chat.subprocess, "Popen", fake_popen)
        chat.speak_blocking("hi", "Samantha")
        assert invocations == [["say", "-v", "Samantha", "--", "hi"]]


# ---- stop_speaking -----------------------------------------------------

class TestStopSpeaking:
    def setup_method(self):
        chat._SAY_PROCESS = None

    def test_noop_when_no_process(self):
        chat.stop_speaking()  # must not raise

    def test_terminates_running_process(self):
        proc = FakeProc("dummy")
        chat._SAY_PROCESS = proc
        chat.stop_speaking()
        assert proc.terminated


# ---- main() argparse routing ------------------------------------------

class TestMainRouting:
    def test_text_mode_routes_to_text_chat(self, monkeypatch):
        called = {}
        monkeypatch.setattr(chat, "text_chat",
                            lambda *a, **kw: called.setdefault("text", (a, kw)))
        monkeypatch.setattr(sys, "argv", ["chat", "--mode", "text"])
        chat.main()
        assert "text" in called

    def test_voice_mode_routes_to_voice_chat(self, monkeypatch):
        called = {}
        monkeypatch.setattr(chat, "voice_chat",
                            lambda *a, **kw: called.setdefault("voice", (a, kw)))
        monkeypatch.setattr(sys, "argv", ["chat", "--mode", "voice"])
        chat.main()
        assert "voice" in called

    def test_coach_mode_routes_to_coach_chat(self, monkeypatch):
        called = {}
        monkeypatch.setattr(chat, "coach_chat",
                            lambda *a, **kw: called.setdefault("coach", (a, kw)))
        monkeypatch.setattr(sys, "argv", ["chat", "--mode", "coach"])
        chat.main()
        assert "coach" in called

    def test_coach_mode_overrides_default_temperature_and_max_tokens(self, monkeypatch):
        captured = {}

        def fake_coach(model, voice, temp, mt):
            captured["temp"] = temp
            captured["max_tokens"] = mt

        monkeypatch.setattr(chat, "coach_chat", fake_coach)
        monkeypatch.setattr(sys, "argv", ["chat", "--mode", "coach"])
        chat.main()
        # Coach mode bumps the chat defaults (0.7, 128) to (0.4, 256).
        assert captured["temp"] == 0.4
        assert captured["max_tokens"] == 256

    def test_coach_mode_respects_explicit_overrides(self, monkeypatch):
        captured = {}

        def fake_coach(model, voice, temp, mt):
            captured["temp"] = temp
            captured["max_tokens"] = mt

        monkeypatch.setattr(chat, "coach_chat", fake_coach)
        monkeypatch.setattr(
            sys, "argv",
            ["chat", "--mode", "coach", "--temperature", "0.9", "--max-tokens", "512"],
        )
        chat.main()
        assert captured["temp"] == 0.9
        assert captured["max_tokens"] == 512

    def test_log_level_flag_propagates(self, monkeypatch):
        import _log
        captured = {}
        monkeypatch.setattr(_log, "configure", lambda level=None: captured.setdefault("level", level))
        monkeypatch.setattr(chat, "text_chat", lambda *a, **kw: None)
        monkeypatch.setattr(sys, "argv", ["chat", "--log-level", "DEBUG", "--mode", "text"])
        chat.main()
        assert captured["level"] == "DEBUG"
