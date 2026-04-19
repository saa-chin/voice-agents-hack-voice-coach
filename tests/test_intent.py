"""Coverage for cli/intent.py — on-device intent routing.

Two classifiers under test:
  - HeuristicClassifier (always available): pattern-matched verdicts.
  - FunctionGemmaClassifier (with a fake Cactus complete fn): structured
    JSON parsing, fallback behaviour on bad replies, and timing fields.

The top-level classify() dispatcher is exercised through both paths so
we lock in the "trust FunctionGemma when confident, else fall back" rule.
"""
from __future__ import annotations

import json

import pytest

import intent as intent_mod
from intent import (
    FunctionGemmaClassifier,
    HeuristicClassifier,
    Intent,
    IntentResult,
    classify,
    _parse_funcgemma_reply,
)


# ---- HeuristicClassifier --------------------------------------------------


@pytest.mark.parametrize(
    "utterance,expected",
    [
        # REST family
        ("I'm tired", Intent.REST),
        ("i am too tired", Intent.REST),
        ("can we stop", Intent.REST),
        ("end the session", Intent.REST),
        ("that's enough for today", Intent.REST),
        ("I'm done", Intent.REST),
        ("rest", Intent.REST),
        ("stop", Intent.REST),
        ("take a break please", Intent.REST),

        # REPEAT family
        ("say it again", Intent.REPEAT),
        ("can you repeat that", Intent.REPEAT),
        ("repeat the prompt", Intent.REPEAT),
        ("one more time", Intent.REPEAT),
        ("what was that", Intent.REPEAT),
        ("I didn't catch that", Intent.REPEAT),
        ("play that back", Intent.REPEAT),
        ("again", Intent.REPEAT),

        # SKIP family
        ("skip this one", Intent.SKIP),
        ("next drill", Intent.SKIP),
        ("move on please", Intent.SKIP),
        ("go to the next", Intent.SKIP),
        ("skip", Intent.SKIP),
        ("I can't do this", Intent.SKIP),

        # NONE: drill-like or off-topic input must NOT match.
        ("aaaaah", Intent.NONE),
        ("the blue spot", Intent.NONE),
        ("one two three four", Intent.NONE),
        ("hello there friend", Intent.NONE),
        ("", Intent.NONE),
        ("   ", Intent.NONE),
    ],
)
def test_heuristic_routes_common_phrases(utterance, expected):
    h = HeuristicClassifier()
    result = h.classify(utterance)
    assert isinstance(result, IntentResult)
    assert result.intent is expected, (
        f"{utterance!r} expected {expected} got {result.intent} "
        f"(confidence={result.confidence})"
    )
    assert result.source == "heuristic"
    assert result.latency_ms >= 0


def test_heuristic_low_confidence_resolves_to_none():
    """Patterns below MIN_CONFIDENCE must NOT trigger an action."""
    # Force a sub-threshold pattern in directly to lock the gate.
    import re
    saved = list(intent_mod._PATTERNS)
    try:
        intent_mod._PATTERNS.clear()
        intent_mod._PATTERNS.append(
            intent_mod._Pattern(Intent.SKIP, re.compile(r"\bmaybe\b"), 0.3)
        )
        h = HeuristicClassifier()
        result = h.classify("well maybe later")
        assert result.intent is Intent.NONE
        # The matching score is reported as the (sub-threshold) confidence
        # so the UI can show "we saw a hint but didn't act".
        assert result.confidence == pytest.approx(0.3)
    finally:
        intent_mod._PATTERNS[:] = saved


def test_heuristic_picks_highest_confidence_when_multiple_match():
    """If two patterns match the same utterance, the higher one wins."""
    h = HeuristicClassifier()
    # "next drill" should beat the more generic "move on" if both fired
    # (they don't overlap, but the principle: explicit > generic).
    result = h.classify("yeah next drill please")
    assert result.intent is Intent.SKIP
    assert result.confidence >= 0.9


def test_heuristic_normalises_input():
    """Whitespace + case shouldn't change the verdict."""
    h = HeuristicClassifier()
    a = h.classify("  I'M TIRED  ")
    b = h.classify("i'm tired")
    assert a.intent is b.intent is Intent.REST
    # Normalised utterance is echoed back lowercase + trimmed.
    assert a.utterance == "i'm tired"


def test_intent_result_payload_shape():
    h = HeuristicClassifier()
    result = h.classify("skip this one")
    payload = result.to_payload()
    assert payload["action"] == "skip"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["utterance"] == "skip this one"
    assert payload["source"] == "heuristic"
    assert isinstance(payload["latency_ms"], int)


# ---- FunctionGemma reply parsing -----------------------------------------


@pytest.mark.parametrize(
    "raw,expected_intent,expected_conf",
    [
        ('{"function":"skip","confidence":0.92}', Intent.SKIP, 0.92),
        ('{"function":"rest","confidence":1.0}', Intent.REST, 1.0),
        ('{"function": "repeat_prompt", "confidence": 0.8}', Intent.REPEAT, 0.8),
        ('{"function":"none","confidence":0.95}', Intent.NONE, 0.95),
        # Tolerates markdown fences:
        ('```json\n{"function":"skip","confidence":0.7}\n```', Intent.SKIP, 0.7),
        # Tolerates leading prose:
        ('Here you go: {"function":"rest","confidence":0.6}', Intent.REST, 0.6),
        # Out-of-range confidence is clamped to [0, 1].
        ('{"function":"skip","confidence":1.5}', Intent.SKIP, 1.0),
        ('{"function":"skip","confidence":-0.2}', Intent.SKIP, 0.0),
        # Missing confidence defaults to 1.0 (clean tool-call match).
        ('{"function":"skip"}', Intent.SKIP, 1.0),
    ],
)
def test_parse_funcgemma_reply_happy_paths(raw, expected_intent, expected_conf):
    parsed = _parse_funcgemma_reply(raw)
    assert parsed is not None
    intent, conf = parsed
    assert intent is expected_intent
    assert conf == pytest.approx(expected_conf)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        "{",
        "{}",
        '{"function":"unknown","confidence":1.0}',
        '{"function":42,"confidence":1.0}',
        '{"confidence":0.9}',
        # Garbage confidence is forgiven (defaults to 1.0) but no `function`
        # is fatal.
        '{"foo":"bar"}',
    ],
)
def test_parse_funcgemma_reply_returns_none_on_garbage(raw):
    assert _parse_funcgemma_reply(raw) is None


def test_parse_funcgemma_reply_handles_non_numeric_confidence():
    """Confidence that isn't a number must NOT crash — defaults to 1.0."""
    parsed = _parse_funcgemma_reply(
        '{"function":"skip","confidence":"high"}'
    )
    assert parsed is not None
    intent, conf = parsed
    assert intent is Intent.SKIP
    assert conf == pytest.approx(1.0)


def test_parse_funcgemma_reply_handles_balanced_garbage():
    """A `{...}` that balances braces but isn't valid JSON returns None."""
    # Trailing comma is invalid JSON in strict mode.
    parsed = _parse_funcgemma_reply('{"function":"skip",}')
    assert parsed is None


def test_parse_funcgemma_reply_extracts_from_array_wrapper():
    """A model that returns [{...}] is still useful — extract the
    first inner object rather than rejecting the whole reply. Lines
    up with how parse_coach_json treats leading prose."""
    parsed = _parse_funcgemma_reply('[{"function":"skip","confidence":0.8}]')
    assert parsed is not None
    intent, conf = parsed
    assert intent is Intent.SKIP
    assert conf == pytest.approx(0.8)


# ---- FunctionGemmaClassifier (with a fake completion fn) -----------------


def _fake_complete(reply: str):
    """Return a callable matching the (messages_json, options_json) shape."""
    def _f(_messages, _options):
        return reply
    return _f


def test_funcgemma_classifier_parses_clean_reply():
    fg = FunctionGemmaClassifier(
        _fake_complete('{"function":"skip","confidence":0.9}')
    )
    result = fg.classify("next one")
    assert result.intent is Intent.SKIP
    assert result.confidence == pytest.approx(0.9)
    assert result.source == "functiongemma"
    assert result.latency_ms >= 0


def test_funcgemma_classifier_falls_through_on_bad_json():
    fg = FunctionGemmaClassifier(_fake_complete("not json"))
    result = fg.classify("skip this")
    assert result.intent is Intent.NONE
    assert result.source == "functiongemma"
    # Raw payload is captured for debugging.
    assert "not json" in result.raw["reply"]


def test_funcgemma_classifier_handles_complete_exception():
    """Any exception from the completion fn must NOT propagate."""
    def _boom(_m, _o):
        raise RuntimeError("kaboom")
    fg = FunctionGemmaClassifier(_boom)
    result = fg.classify("skip")
    assert result.intent is Intent.NONE
    assert result.raw == {"error": "kaboom"}


def test_funcgemma_classifier_short_circuits_on_empty_input():
    """No model call for empty input — saves the user a wasted turn."""
    calls = []
    def _f(messages, options):
        calls.append(messages)
        return '{"function":"skip","confidence":1.0}'
    fg = FunctionGemmaClassifier(_f)
    result = fg.classify("   ")
    assert result.intent is Intent.NONE
    assert calls == [], "should not have called the model on empty utterance"


def test_funcgemma_classifier_passes_utterance_through_prompt():
    """The utterance must reach the model in the system prompt."""
    seen = {}
    def _f(messages_json, _options):
        seen["messages"] = json.loads(messages_json)
        return '{"function":"none","confidence":0.9}'
    fg = FunctionGemmaClassifier(_f)
    fg.classify("end the session please")
    assert seen["messages"][0]["role"] == "system"
    assert "end the session please" in seen["messages"][0]["content"]


# ---- classify() dispatcher -----------------------------------------------


def test_classify_uses_heuristic_when_no_funcgemma():
    result = classify("I'm tired", funcgemma=None)
    assert result.intent is Intent.REST
    assert result.source == "heuristic"


def test_classify_prefers_confident_funcgemma_verdict():
    fg = FunctionGemmaClassifier(
        _fake_complete('{"function":"rest","confidence":0.95}')
    )
    result = classify("anything goes here", funcgemma=fg)
    assert result.intent is Intent.REST
    assert result.source == "functiongemma"


def test_classify_falls_back_when_funcgemma_returns_low_confidence_action():
    """A low-confidence non-NONE verdict should NOT win — let the heuristic try."""
    fg = FunctionGemmaClassifier(
        _fake_complete('{"function":"skip","confidence":0.2}')
    )
    # Heuristic will catch this clearly.
    result = classify("I'm tired", funcgemma=fg)
    assert result.intent is Intent.REST
    assert result.source == "heuristic"


def test_classify_respects_funcgemma_confident_none():
    """High-confidence NONE means the user clearly didn't say a command —
    don't second-guess with a regex."""
    fg = FunctionGemmaClassifier(
        _fake_complete('{"function":"none","confidence":0.95}')
    )
    # Even though 'rest' is in the string, the confident NONE wins.
    result = classify("I had a good rest yesterday", funcgemma=fg)
    assert result.intent is Intent.NONE
    assert result.source == "functiongemma"


def test_classify_falls_back_when_funcgemma_uncertain_none():
    """Low-confidence NONE means 'I'm not sure' — let the heuristic try."""
    fg = FunctionGemmaClassifier(
        _fake_complete('{"function":"none","confidence":0.4}')
    )
    result = classify("skip this one", funcgemma=fg)
    assert result.intent is Intent.SKIP
    assert result.source == "heuristic"
