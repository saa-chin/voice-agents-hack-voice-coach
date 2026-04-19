"""Coverage for cli/coach.py pure helpers and session-log integration.

Excluded from these tests (they require real hardware / native lib):
  - capture_one_utterance() — needs sounddevice + a microphone
  - coach_mode() — full integration, needs Cactus + Gemma weights

Both are exercised by hand via `./run-cli --coach` and indirectly by the
WebSocket happy-path test in test_server.py with Cactus mocked.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import coach
import content
import pytest


# ---- _extract_first_json_object ------------------------------------------

class TestExtractFirstJsonObject:
    def test_bare_object(self):
        out = coach._extract_first_json_object('{"a": 1}')
        assert out == {"a": 1}

    def test_with_leading_prose(self):
        out = coach._extract_first_json_object("sure thing: {\"a\": 1} cheers")
        assert out == {"a": 1}

    def test_markdown_fence_json(self):
        raw = "```json\n{\"a\": 1}\n```"
        out = coach._extract_first_json_object(raw)
        assert out == {"a": 1}

    def test_markdown_fence_bare(self):
        raw = "```\n{\"a\": 1}\n```"
        out = coach._extract_first_json_object(raw)
        assert out == {"a": 1}

    def test_nested_objects_balanced(self):
        out = coach._extract_first_json_object('{"a": {"b": {"c": 3}}}')
        assert out == {"a": {"b": {"c": 3}}}

    def test_no_braces(self):
        assert coach._extract_first_json_object("hello world") is None

    def test_empty_string(self):
        assert coach._extract_first_json_object("") is None

    def test_unbalanced_braces(self):
        assert coach._extract_first_json_object("{{{") is None

    def test_malformed_json_inside_balanced_braces(self):
        # Balanced braces but invalid JSON content.
        out = coach._extract_first_json_object("{not json}")
        assert out is None

    def test_picks_first_object_only(self):
        raw = '{"first": 1} {"second": 2}'
        out = coach._extract_first_json_object(raw)
        assert out == {"first": 1}


# ---- parse_coach_json (with Cactus envelope unwrap) ----------------------

class TestParseCoachJson:
    @pytest.fixture
    def valid_inner(self):
        return {
            "ack": "Nice tone",
            "feedback": "Try one more time, a little louder.",
            "next_action": "retry",
            "metrics_observed": {"matched_prompt": True, "loudness_ok": False},
        }

    def test_bare_json(self, valid_inner):
        out = coach.parse_coach_json(json.dumps(valid_inner))
        assert out == valid_inner

    def test_unwraps_cactus_envelope(self, valid_inner):
        envelope = {
            "success": True,
            "error": None,
            "cloud_handoff": False,
            "response": json.dumps(valid_inner),
        }
        out = coach.parse_coach_json(json.dumps(envelope))
        assert out == valid_inner

    def test_envelope_with_indented_inner(self, valid_inner):
        envelope = {"success": True, "response": json.dumps(valid_inner, indent=2)}
        out = coach.parse_coach_json(json.dumps(envelope))
        assert out == valid_inner

    def test_envelope_with_nonjson_response_returns_envelope(self):
        envelope = {"success": True, "response": "not json at all"}
        out = coach.parse_coach_json(json.dumps(envelope))
        # Falls back to returning the envelope dict itself.
        assert out == envelope

    def test_does_not_unwrap_if_schema_keys_present_at_top(self, valid_inner):
        """If the top-level already has our schema, don't unwrap into 'response'."""
        wrapped = dict(valid_inner)
        wrapped["response"] = "ignore me"
        wrapped["success"] = True
        out = coach.parse_coach_json(json.dumps(wrapped))
        assert out["ack"] == valid_inner["ack"]

    def test_garbage_returns_none(self):
        assert coach.parse_coach_json("definitely not json") is None

    def test_empty_returns_none(self):
        assert coach.parse_coach_json("") is None

    def test_normalises_compact_field_names(self):
        """The slim COACH_SESSION_PREAMBLE asks Gemma to emit compact
        keys (h/m/a/f/n) so we burn fewer decode tokens. The parser
        must promote them to the canonical long names so the rest of
        the pipeline (validator, enforcer, WS payload) keeps working."""
        compact = {
            "h": "ah",
            "m": 1,
            "a": "Strong voice.",
            "f": "Same energy on the next one.",
            "n": "advance",
        }
        out = coach.parse_coach_json(json.dumps(compact))
        assert out["heard"] == "ah"
        assert out["ack"] == "Strong voice."
        assert out["feedback"] == "Same energy on the next one."
        assert out["next_action"] == "advance"
        assert out["metrics_observed"]["matched_prompt"] is True
        # Compact keys gone — no duplicate fields.
        for k in ("h", "m", "a", "f", "n"):
            assert k not in out

    def test_compact_m_zero_maps_to_matched_prompt_false(self):
        """m=0 (mismatch) must reach the strict-matching enforcer as
        matched_prompt=False so a praise-y ack still gets scrubbed."""
        compact = {
            "h": "thank you",
            "m": 0,
            "a": "Great job!",
            "f": "moving on",
            "n": "advance",
        }
        out = coach.validate_coach_json(coach.parse_coach_json(json.dumps(compact)))
        assert out is not None
        # Enforcer kicked in: action flipped to retry, praise scrubbed.
        assert out["next_action"] == "retry"
        assert "great job" not in out["ack"].lower()

    def test_compact_keys_inside_cactus_envelope(self):
        """Envelope unwrap + compact-key normalisation must compose."""
        envelope = {
            "success": True,
            "response": json.dumps({
                "h": "ah", "m": 1, "a": "Clear and full.",
                "f": "Same energy on the next one.", "n": "advance",
            }),
        }
        out = coach.parse_coach_json(json.dumps(envelope))
        assert out["heard"] == "ah"
        assert out["ack"] == "Clear and full."
        assert out["next_action"] == "advance"


# ---- validate_coach_json --------------------------------------------------

class TestValidateCoachJson:
    def test_full_valid(self):
        obj = {
            "ack": "ok", "feedback": "fb",
            "next_action": "advance",
            "metrics_observed": {"loudness_ok": True},
        }
        out = coach.validate_coach_json(obj)
        assert out is obj
        assert out["metrics_observed"] == {"loudness_ok": True}

    def test_missing_ack_rejected(self):
        assert coach.validate_coach_json({"feedback": "x", "next_action": "advance"}) is None

    def test_missing_feedback_rejected(self):
        assert coach.validate_coach_json({"ack": "x", "next_action": "advance"}) is None

    def test_missing_next_action_rejected(self):
        assert coach.validate_coach_json({"ack": "x", "feedback": "y"}) is None

    @pytest.mark.parametrize("action", ["retry", "advance", "rest"])
    def test_valid_next_action(self, action):
        obj = {"ack": "a", "feedback": "b", "next_action": action}
        assert coach.validate_coach_json(obj) is obj

    @pytest.mark.parametrize("action", ["sing", "stop", "RETRY", "", None, 5])
    def test_invalid_next_action_rejected(self, action):
        obj = {"ack": "a", "feedback": "b", "next_action": action}
        assert coach.validate_coach_json(obj) is None

    def test_metrics_observed_defaulted_when_missing(self):
        obj = {"ack": "a", "feedback": "b", "next_action": "rest"}
        out = coach.validate_coach_json(obj)
        assert out is not None
        assert out["metrics_observed"] == {}

    def test_none_input_rejected(self):
        assert coach.validate_coach_json(None) is None  # type: ignore[arg-type]

    def test_empty_dict_rejected(self):
        assert coach.validate_coach_json({}) is None


# ---- _enforce_strict_matching --------------------------------------------
#
# The enforcer is the server-side guard that catches the demo bug the
# whole rewrite was triggered by: "I said something completely different
# and the coach still said 'good job'." Even when the model returns
# matched_prompt=false, it sometimes pairs that with next_action=advance
# and a praise-y ack. The enforcer overrides those inconsistencies so
# the patient is never told they were correct when they weren't.

class TestStrictMatchingEnforcer:
    def _base(self, **overrides):
        obj = {
            "heard": "thank you",
            "ack": "Nice try!",
            "feedback": "Good attempt — keep going.",
            "next_action": "advance",
            "metrics_observed": {"matched_prompt": False, "loudness_ok": True},
        }
        obj.update(overrides)
        return obj

    def test_forces_retry_when_matched_prompt_false(self):
        obj = self._base(next_action="advance")
        out = coach._enforce_strict_matching(obj)
        assert out["next_action"] == "retry"

    def test_does_not_touch_action_when_matched_prompt_true(self):
        obj = self._base(
            metrics_observed={"matched_prompt": True, "loudness_ok": True},
            next_action="advance",
        )
        out = coach._enforce_strict_matching(obj)
        assert out["next_action"] == "advance"

    def test_scrubs_praise_from_ack_on_mismatch(self):
        obj = self._base(ack="Nice try!", heard="thank you")
        out = coach._enforce_strict_matching(obj)
        assert "nice" not in out["ack"].lower()
        # Honest replacement names what was actually heard.
        assert "thank you" in out["ack"].lower()

    @pytest.mark.parametrize("praise", [
        "Nice try!", "Great attempt!", "Good job!", "Wonderful!",
        "Excellent!", "Well done!", "Fantastic!", "Perfect!", "Awesome!",
        "Beautiful!", "Lovely!", "Bravo!", "Way to go!",
        "You got it!", "Keep it up!",
    ])
    def test_strips_known_praise_vocab_on_mismatch(self, praise):
        obj = self._base(ack=praise, heard="thank you")
        out = coach._enforce_strict_matching(obj)
        assert not coach._looks_like_praise(out["ack"]), (
            f"praise leaked through: {out['ack']!r}"
        )

    def test_uses_neutral_ack_when_nothing_was_heard(self):
        obj = self._base(ack="Great effort!", heard="(nothing clear)")
        out = coach._enforce_strict_matching(obj)
        assert "didn't catch" in out["ack"].lower()

    def test_uses_fallback_ack_when_heard_is_blank(self):
        obj = self._base(ack="Wonderful!", heard="")
        out = coach._enforce_strict_matching(obj)
        # No transcript to quote — falls back to a generic neutral line.
        assert "not quite" in out["ack"].lower()

    def test_does_not_scrub_legitimate_positive_ack_on_match(self):
        """When matched_prompt=true, 'Strong voice' / 'Clear and full'
        type acks are LEGITIMATE — the enforcer must leave them alone."""
        obj = self._base(
            metrics_observed={"matched_prompt": True, "loudness_ok": True},
            ack="Nice and full.",  # contains 'nice' but is correct praise
            next_action="advance",
        )
        out = coach._enforce_strict_matching(obj)
        assert out["ack"] == "Nice and full."
        assert out["next_action"] == "advance"

    def test_validate_coach_json_runs_enforcer(self):
        """End-to-end: a praise+advance reply with matched_prompt=false
        comes out the other side as retry + neutral ack."""
        raw = {
            "heard": "thank you",
            "ack": "Great job!",
            "feedback": "moving on",
            "next_action": "advance",
            "metrics_observed": {"matched_prompt": False},
        }
        out = coach.validate_coach_json(raw)
        assert out is not None
        assert out["next_action"] == "retry"
        assert "great job" not in out["ack"].lower()


class TestPraiseVocabulary:
    """Direct probe of _looks_like_praise — the predicate that drives
    whether the enforcer rewrites an ack. False positives here would
    nuke legitimate feedback, so the bar for matching is intentionally
    word-bounded."""

    @pytest.mark.parametrize("text", [
        "Nice try", "great", "Good job", "wonderful effort",
        "you got it", "Way to go", "excellent",
        "Perfect.", "amazing!", "Lovely.",
    ])
    def test_detects_known_praise_words(self, text):
        assert coach._looks_like_praise(text)

    @pytest.mark.parametrize("text", [
        "I heard 'thank you'.",
        "The prompt was 'Help'.",
        "Take a breath, then say it again.",
        "Strong voice.",            # legitimate match feedback
        "Clear and full.",          # legitimate match feedback
        "I caught a hesitation.",
        "Got silence.",
        "",
    ])
    def test_does_not_flag_neutral_or_legitimate_text(self, text):
        assert not coach._looks_like_praise(text)


# ---- rms_to_dbfs ---------------------------------------------------------

class TestRmsToDbfs:
    def test_zero_is_negative_infinity(self):
        assert coach.rms_to_dbfs(0.0) == -math.inf

    def test_negative_treated_as_zero(self):
        assert coach.rms_to_dbfs(-5.0) == -math.inf

    def test_full_scale_is_zero(self):
        assert coach.rms_to_dbfs(32768.0) == pytest.approx(0.0, abs=1e-6)

    def test_half_scale_is_minus_six(self):
        # 20*log10(0.5) ≈ -6.02 dB
        assert coach.rms_to_dbfs(16384.0) == pytest.approx(-6.02, abs=0.01)

    def test_typical_speech_value(self):
        # Quiet speech around RMS=2000 → ~-24 dBFS
        v = coach.rms_to_dbfs(2000.0)
        assert -25.0 < v < -23.0


# ---- build_prompt --------------------------------------------------------

class TestBuildPrompt:
    @pytest.fixture
    def warmup_drill(self):
        # The current lesson library is main-task-only, so we hand-build a
        # minimal warmup drill with no focus / no note — preserving coverage
        # of the "instruction-only phase" rendering path.
        return content.Drill(
            stage="warmup",
            index=0,
            prompt="Take a deep breath and sustain 'Ahhh'.",
            note="",
            target_dbfs=content.DEFAULT_TARGET_DBFS_BY_STAGE["warmup"],
            exercise_name="Long AH",
            focus="",
        )

    @pytest.fixture
    def main_task_drill(self):
        # Use a real LSVT functional-phrase drill so the prompt builder
        # is exercised against a realistic main-task line.
        drills = content.drills_for_exercise("L_func_basics")
        return next(d for d in drills if d.stage == "main_task")

    def test_includes_drill_prompt_text(self, warmup_drill):
        out = coach.build_prompt(warmup_drill, -22.0, 1.5)
        assert warmup_drill.prompt in out

    def test_uses_human_friendly_stage_label(self, main_task_drill):
        out = coach.build_prompt(main_task_drill, -15.0, 5.0)
        # main_task → "main task" (underscore stripped) in the rendered prompt
        assert "main task" in out
        # The raw enum name should NOT appear as user-visible label text
        assert "main_task" not in out

    def test_includes_measurements(self, main_task_drill):
        out = coach.build_prompt(main_task_drill, -19.4, 2.31)
        assert "-19.4" in out
        assert "2.3" in out
        assert f"{main_task_drill.target_dbfs}" in out

    def test_handles_negative_infinity_dbfs(self, warmup_drill):
        """Silence (-inf dBFS) must not produce 'inf' in the prompt — Gemma chokes on that."""
        out = coach.build_prompt(warmup_drill, -math.inf, 0.0)
        assert "inf" not in out.lower()
        assert "-90" in out

    def test_mentions_json_only(self, warmup_drill):
        """Slim prompt MUST still document the schema. We accept either
        the canonical long field names OR the compact aliases (h/m/a/f/n)
        — the model is free to emit either, the parser normalises both."""
        out = coach.build_prompt(warmup_drill, -20.0, 1.0)
        assert "JSON" in out
        # Compact field names are present in COACH_SESSION_PREAMBLE.
        for key in ("h", "m", "a", "f", "n"):
            assert f'"{key}"' in out
        # And the canonical long names appear in the explanation block
        # so the model knows what they map to.
        for key in ("heard", "ack", "feedback", "next_action", "matched_prompt"):
            assert key in out

    def test_includes_exercise_name(self, warmup_drill):
        out = coach.build_prompt(warmup_drill, -20.0, 1.0)
        # `warmup_drill` is the LSVT canonical sustained-AH exercise.
        assert "Long AH" in out

    def test_unknown_stage_label_falls_back_gracefully(self):
        """Custom drills with unknown stages should still render readable prompts."""
        d = content.Drill(stage="custom_stage", index=0, prompt="hello",
                          exercise_name="Custom")
        out = coach.build_prompt(d, -20.0, 1.0)
        # Underscores get replaced with spaces in the displayed label.
        assert "custom stage" in out

    def test_session_preamble_is_constant(self):
        """The session preamble must NOT vary by drill — that's what
        lets us pin it in Cactus's KV cache once per session and only
        prefill the small per-drill delta on subsequent turns."""
        a = content.Drill(stage="warmup", index=0, prompt="ah",
                          exercise_name="One")
        b = content.Drill(stage="main_task", index=3, prompt="help",
                          exercise_name="Two")
        # Same preamble ID, regardless of drill.
        out_a = coach.build_prompt(a, -20.0, 1.0)
        out_b = coach.build_prompt(b, -15.0, 2.5)
        # Both contain the preamble text byte-for-byte.
        assert coach.COACH_SESSION_PREAMBLE in out_a
        assert coach.COACH_SESSION_PREAMBLE in out_b

    def test_drill_prompt_is_small(self):
        """Per-turn delta must be tiny — that's the whole point of the
        split. Cap is a soft 400 chars (vs the old 3 000-token system
        prompt). Bumps need a deliberate reason."""
        d = content.Drill(stage="main_task", index=0, prompt="Help",
                          exercise_name="LSVT functional")
        delta = coach.build_drill_prompt(d, -19.0, 2.0)
        assert len(delta) < 400, f"per-drill delta grew to {len(delta)} chars"


# ---- _join_for_speech ----------------------------------------------------

class TestJoinForSpeech:
    def test_strips_trailing_punctuation_and_joins(self):
        out = coach._join_for_speech("That was good.", "Try once more.")
        # No double periods.
        assert ".." not in out
        assert out == "That was good. Try once more."

    def test_handles_already_unpunctuated(self):
        out = coach._join_for_speech("Nice", "Hold longer")
        assert out == "Nice. Hold longer."

    def test_strips_mixed_trailing_punct(self):
        out = coach._join_for_speech("Wow!", "Keep going,,, ")
        assert out == "Wow. Keep going."

    def test_empty_ack(self):
        out = coach._join_for_speech("", "Just feedback.")
        assert out == "Just feedback."

    def test_empty_feedback(self):
        out = coach._join_for_speech("Just ack.", "")
        assert out == "Just ack."

    def test_both_empty(self):
        assert coach._join_for_speech("", "") == ""

    def test_whitespace_only(self):
        assert coach._join_for_speech("   ", "\t\n") == ""


# ---- _TokenCollector ------------------------------------------------------

class TestTokenCollector:
    def test_accumulates_tokens(self):
        c = coach._TokenCollector()
        c("hello", 1)
        c(" ", 2)
        c("world", 3)
        assert c.text() == "hello world"

    def test_ignores_empty_strings(self):
        c = coach._TokenCollector()
        c("", 1)
        c(None, 2)  # type: ignore[arg-type]  # defensive
        c("ok", 3)
        assert c.text() == "ok"

    def test_empty_collector_returns_empty_string(self):
        assert coach._TokenCollector().text() == ""

    def test_callable_signature(self):
        """Must be invocable as a function: cb(text, token_id)."""
        c = coach._TokenCollector()
        assert callable(c)
        c("a", 0)
        c("b", 1)
        assert c.text() == "ab"


# ---- session log round-trip ---------------------------------------------

class TestSessionLog:
    def test_open_session_log_creates_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(coach, "SESSION_DIR", tmp_path / "nested" / "sessions")
        path = coach._open_session_log()
        assert path.parent.exists()
        assert path.name.startswith("session-") and path.name.endswith(".jsonl")

    def test_append_jsonl_writes_a_line_per_call(self, tmp_path):
        path = tmp_path / "session.jsonl"
        coach._append_jsonl(path, {"turn": 1, "ack": "nice"})
        coach._append_jsonl(path, {"turn": 2, "ack": "again"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"turn": 1, "ack": "nice"}
        assert json.loads(lines[1]) == {"turn": 2, "ack": "again"}

    def test_append_jsonl_unwritable_path_does_not_raise(self, tmp_path, caplog):
        # Direct write into a non-existent parent directory.
        bad = tmp_path / "no" / "such" / "dir" / "x.jsonl"
        # Should warn but not raise — we never want session logging to crash a turn.
        coach._append_jsonl(bad, {"x": 1})

    def test_append_jsonl_unicode_safe(self, tmp_path):
        path = tmp_path / "u.jsonl"
        coach._append_jsonl(path, {"heard": "café — naïve"})
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["heard"] == "café — naïve"


# ---- module-level constants are sane ------------------------------------

def test_audio_constants_are_consistent():
    """The frame-derived constants must agree with each other."""
    assert coach.FRAME_SAMPLES == coach.SR * coach.FRAME_MS // 1000
    assert coach.SR == 16000
    assert 0 < coach.FRAME_MS < 200
    assert coach.MAX_UTTERANCE_MS > coach.MIN_SPEECH_MS
    assert coach.MAX_RETRIES_PER_DRILL >= 1
