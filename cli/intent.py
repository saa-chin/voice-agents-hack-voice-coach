"""On-device intent routing for Voice Coach.

Free-form patient utterances ("I'm tired", "say it again", "next one")
are mapped to existing in-app actions (rest, repeat_prompt, skip) by a
small specialist model so the heavy Gemma 4 coach is never woken up for
control-flow decisions.

Two layers, in order of preference:

  1. FunctionGemmaClassifier — wraps FunctionGemma 270M via Cactus.
     Built for tool-calling style structured output. Sub-100 ms on-device.

  2. HeuristicClassifier — a regex / keyword fallback. Always available
     even when the dylib or FunctionGemma weights aren't loaded yet, so
     the UX never blocks on model load and tests run without Cactus.

Both classifiers return the same `IntentResult` so callers (the WS
handler) don't care which one ran. `classify()` picks the model when
available and falls back to the heuristic otherwise.

The action vocabulary is intentionally tiny and matches the existing
`{ type: "command", action: ... }` WS contract. Adding a new intent
means: extend `Intent`, add patterns to the heuristic, extend the
function-call schema in `FUNCGEMMA_SYSTEM_TEMPLATE`, and wire the
new action through `web-py/backend/app/main.py`.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import _log

log = _log.get("intent")


# Default FunctionGemma model id. Overridable via env var so the demo
# can swap to a larger function-calling model without touching code.
DEFAULT_FUNCGEMMA_ID = "google/functiongemma-270m-it"


class Intent(str, Enum):
    """Closed set of actions the router can dispatch.

    Values match the `action` strings already accepted by the WS
    `command` message, so the WS handler can forward them without
    translation. NONE means "no confident match — do nothing".
    """

    SKIP = "skip"
    REST = "rest"
    REPEAT = "repeat_prompt"
    NONE = "none"


@dataclass
class IntentResult:
    """One classification verdict.

    Attributes:
        intent: Routed action (or NONE).
        confidence: 0..1 self-reported confidence. Heuristic emits
            calibrated values per pattern; FunctionGemma echoes its
            own confidence field if produced, else 1.0 on a clean
            tool-call match.
        utterance: The original input text, normalised.
        source: "functiongemma" or "heuristic" — surfaced to the
            client so the UI can label which model decided.
        latency_ms: Wall-clock time spent in classify(), rounded.
        raw: Optional debug payload (e.g. the model's raw reply).
    """

    intent: Intent
    confidence: float
    utterance: str
    source: str
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Wire-friendly dict for the WS `intent_result` frame."""
        return {
            "action": self.intent.value,
            "confidence": round(self.confidence, 3),
            "utterance": self.utterance,
            "source": self.source,
            "latency_ms": self.latency_ms,
        }


# --- Heuristic classifier --------------------------------------------------
#
# The patterns below are not exhaustive — they're the high-frequency
# phrases an older patient with dysarthria is most likely to use, drawn
# from LSVT LOUD coaching transcripts and the literature on between-
# session communication breakdown. Each pattern carries a confidence
# weight; we pick the highest match and bail out if nothing scores
# above MIN_CONFIDENCE (so genuine drill speech doesn't get hijacked).

# Confidence below which we refuse to act. Keeps "I had coffee earlier"
# from being read as REST just because it contains "had".
MIN_CONFIDENCE = 0.55


# Order matters only for ties — confidence resolves first. Patterns are
# match-anywhere by default; anchor with ^ / $ to be strict. We use
# word boundaries on the keyword to avoid "request" matching "rest".
@dataclass(frozen=True)
class _Pattern:
    intent: Intent
    regex: re.Pattern[str]
    confidence: float


def _build_patterns() -> list[_Pattern]:
    return [
        # ---- REST ----------------------------------------------------------
        _Pattern(Intent.REST, re.compile(r"\b(i\s*am|i'?m)\s+(too\s+)?tired\b"), 0.95),
        _Pattern(Intent.REST, re.compile(r"\bi\s+need\s+(a\s+)?(rest|break)\b"), 0.95),
        _Pattern(Intent.REST, re.compile(r"\b(let'?s|can\s+we)\s+(stop|rest|finish|end)\b"), 0.9),
        _Pattern(Intent.REST, re.compile(r"\b(end|stop|finish|quit|exit)\s+(the\s+)?session\b"), 0.95),
        _Pattern(Intent.REST, re.compile(r"\bi'?m\s+done\b"), 0.85),
        _Pattern(Intent.REST, re.compile(r"\b(that'?s\s+enough|enough\s+for\s+today)\b"), 0.9),
        _Pattern(Intent.REST, re.compile(r"\btake\s+a\s+break\b"), 0.85),
        # Single keyword: lower confidence so a stray "rest" inside a
        # phrase doesn't trip it; "rest" alone as a command still passes.
        _Pattern(Intent.REST, re.compile(r"^\s*(rest|stop|done|quit)\s*[.!?]*\s*$"), 0.8),

        # ---- REPEAT --------------------------------------------------------
        _Pattern(Intent.REPEAT, re.compile(r"\b(say|read|play)\s+(it|that|the\s+prompt)\s+again\b"), 0.95),
        _Pattern(Intent.REPEAT, re.compile(r"\b(can\s+you\s+)?repeat\s+(it|that|the\s+prompt)?\b"), 0.9),
        _Pattern(Intent.REPEAT, re.compile(r"\bone\s+more\s+time\b"), 0.85),
        _Pattern(Intent.REPEAT, re.compile(r"\bwhat\s+(was|is)\s+(it|that|the\s+prompt)\b"), 0.85),
        _Pattern(Intent.REPEAT, re.compile(r"\bi\s+didn'?t\s+(hear|catch)\s+(that|it)\b"), 0.9),
        _Pattern(Intent.REPEAT, re.compile(r"\bplay\s+(it|that)\s+back\b"), 0.85),
        _Pattern(Intent.REPEAT, re.compile(r"^\s*(repeat|again)\s*[.!?]*\s*$"), 0.8),

        # ---- SKIP ----------------------------------------------------------
        _Pattern(Intent.SKIP, re.compile(r"\b(skip|pass|move\s+on|move\s+along|next)\s+(this|that|drill|one|please)?\b"), 0.9),
        _Pattern(Intent.SKIP, re.compile(r"\b(go|move)\s+to\s+the\s+next\b"), 0.9),
        _Pattern(Intent.SKIP, re.compile(r"\bnext\s+(drill|one|prompt|please)\b"), 0.95),
        _Pattern(Intent.SKIP, re.compile(r"\b(let'?s\s+)?move\s+on\b"), 0.85),
        _Pattern(Intent.SKIP, re.compile(r"\bi\s+(can'?t|cannot)\s+(do|say)\s+(this|that|it)\b"), 0.8),
        _Pattern(Intent.SKIP, re.compile(r"^\s*(skip|next|pass)\s*[.!?]*\s*$"), 0.85),
    ]


_PATTERNS: list[_Pattern] = _build_patterns()


def _normalise(utterance: str) -> str:
    """Lowercase + collapse whitespace. Punctuation stays for regex anchors."""
    return re.sub(r"\s+", " ", utterance.strip().lower())


class HeuristicClassifier:
    """Regex/keyword classifier — always available, no model needed."""

    name = "heuristic"

    def classify(self, utterance: str) -> IntentResult:
        t0 = time.monotonic()
        norm = _normalise(utterance)
        best: tuple[Intent, float] | None = None
        if norm:
            for pat in _PATTERNS:
                if pat.regex.search(norm):
                    if best is None or pat.confidence > best[1]:
                        best = (pat.intent, pat.confidence)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if best is None or best[1] < MIN_CONFIDENCE:
            return IntentResult(
                intent=Intent.NONE,
                confidence=best[1] if best else 0.0,
                utterance=norm,
                source=self.name,
                latency_ms=latency_ms,
            )
        return IntentResult(
            intent=best[0],
            confidence=best[1],
            utterance=norm,
            source=self.name,
            latency_ms=latency_ms,
        )


# --- FunctionGemma classifier ---------------------------------------------
#
# We don't use a true tool-call API surface (Cactus exposes raw text
# completion). Instead we prompt FunctionGemma to emit a single JSON
# object that mimics a function call:
#
#   { "function": "rest" | "skip" | "repeat_prompt" | "none",
#     "confidence": 0.0..1.0 }
#
# The 270M model is small enough that a tight, schema-first prompt
# produces clean JSON in well under 100 ms once the model is warm.

FUNCGEMMA_SYSTEM_TEMPLATE = """You are an intent router for a speech-therapy practice app.
The user is in the middle of a drill and just said something to the app.
Decide which one of these app functions, if any, the user wants to call:

- skip            : skip the current drill and move to the next one
- rest            : end the practice session because the user wants to stop or is tired
- repeat_prompt   : read the current drill prompt out loud again
- none            : the user did not request any of the above (e.g. they
                    were attempting the drill itself, or said something
                    off-topic that doesn't match a function)

Reply with ONE JSON object and nothing else. No prose, no markdown.
Schema:
{
  "function": "skip" | "rest" | "repeat_prompt" | "none",
  "confidence": <number between 0 and 1>
}

User utterance:
"""


_VALID_FUNCTIONS = {i.value for i in Intent}


def _parse_funcgemma_reply(raw: str) -> tuple[Intent, float] | None:
    """Pull the function-call JSON out of the model's reply.

    Tolerates the same envelopes coach.parse_coach_json deals with
    (markdown fences, leading prose). Returns None on any malformed
    reply so the caller can fall back to the heuristic.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n\t")
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    blob: str | None = None
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start : i + 1]
                break
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as exc:
        log.warning("intent JSON parse failed: %s; blob=%r", exc, blob[:200])
        return None
    if not isinstance(obj, dict):
        return None
    fn = str(obj.get("function", "")).strip().lower()
    if fn not in _VALID_FUNCTIONS:
        log.debug("intent function out of vocabulary: %r", fn)
        return None
    try:
        conf = float(obj.get("confidence", 1.0))
    except (TypeError, ValueError):
        conf = 1.0
    conf = max(0.0, min(1.0, conf))
    return Intent(fn), conf


class FunctionGemmaClassifier:
    """Cactus-backed FunctionGemma 270M intent router.

    The classifier is constructed lazily by the WS server once the
    model finishes loading. Tests inject a fake `complete_fn` so the
    schema/parse path runs without the real dylib.
    """

    name = "functiongemma"

    def __init__(
        self,
        complete_fn: Callable[[str, str], str],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> None:
        # complete_fn(messages_json, options_json) -> raw reply text.
        # It MUST be safe to call from a worker thread; the server uses
        # asyncio.to_thread() to keep the event loop responsive.
        self._complete = complete_fn
        self._options_json = json.dumps({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "confidence_threshold": 0.0,
        })

    def classify(self, utterance: str) -> IntentResult:
        norm = _normalise(utterance)
        t0 = time.monotonic()
        if not norm:
            return IntentResult(
                intent=Intent.NONE,
                confidence=0.0,
                utterance=norm,
                source=self.name,
                latency_ms=0,
            )
        messages = [
            {"role": "system", "content": FUNCGEMMA_SYSTEM_TEMPLATE + norm},
            {"role": "user", "content": norm},
        ]
        try:
            raw = self._complete(json.dumps(messages), self._options_json)
        except Exception as exc:
            log.warning("FunctionGemma call failed: %s — falling back", exc)
            return IntentResult(
                intent=Intent.NONE,
                confidence=0.0,
                utterance=norm,
                source=self.name,
                latency_ms=int((time.monotonic() - t0) * 1000),
                raw={"error": str(exc)},
            )
        latency_ms = int((time.monotonic() - t0) * 1000)
        parsed = _parse_funcgemma_reply(raw)
        if parsed is None:
            return IntentResult(
                intent=Intent.NONE,
                confidence=0.0,
                utterance=norm,
                source=self.name,
                latency_ms=latency_ms,
                raw={"reply": raw[:200]},
            )
        intent, conf = parsed
        return IntentResult(
            intent=intent,
            confidence=conf,
            utterance=norm,
            source=self.name,
            latency_ms=latency_ms,
            raw={"reply": raw[:200]},
        )


# --- Top-level dispatcher --------------------------------------------------


_HEURISTIC = HeuristicClassifier()


def classify(
    utterance: str,
    funcgemma: FunctionGemmaClassifier | None = None,
) -> IntentResult:
    """Classify a free-form utterance, preferring FunctionGemma when ready.

    If `funcgemma` is provided AND it returns a confident verdict, that
    wins. Otherwise we fall through to the heuristic so the demo keeps
    working without the model loaded (and so a misfiring 270M doesn't
    silently swallow obvious commands).
    """
    if funcgemma is not None:
        result = funcgemma.classify(utterance)
        # Trust the model when it picked a real action with reasonable
        # confidence. NONE-with-low-confidence is just "I'm not sure" —
        # let the heuristic try too in case it spots an obvious keyword.
        if result.intent is not Intent.NONE and result.confidence >= MIN_CONFIDENCE:
            return result
        # If the model is highly confident this is NOT a command (e.g. the
        # user is mid-drill and just said "the blue spot"), respect that
        # and don't second-guess with the regex.
        if result.intent is Intent.NONE and result.confidence >= 0.85:
            return result
    return _HEURISTIC.classify(utterance)


__all__ = [
    "Intent",
    "IntentResult",
    "HeuristicClassifier",
    "FunctionGemmaClassifier",
    "classify",
    "DEFAULT_FUNCGEMMA_ID",
    "MIN_CONFIDENCE",
    "FUNCGEMMA_SYSTEM_TEMPLATE",
]
