"""Coach mode: drill-driven, JSON-structured speech practice.

Per-drill flow:
  1. Speak the prompt aloud (`say`).
  2. Capture one utterance from the mic (energy-VAD, same params as voice chat).
  3. Compute average loudness (dBFS) over voiced frames.
  4. Send audio + a coaching system prompt to Gemma 4.
  5. Parse the JSON reply: { ack, feedback, next_action, metrics_observed }.
  6. Speak `ack + feedback`, append the turn to a session log, follow next_action.

Session JSONL is written to:
  $VOICE_COACH_SESSION_DIR (default: ~/.voice-coach/sessions/)

This module reuses the audio FFI helper and TTS wrappers from chat.py
rather than reimplementing them. It is imported by chat.py when
`--mode coach` is selected; it is not meant to be run directly.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import _log
from _exit import ExitCode, die
from _tts import split_sentences
from content import Drill, default_drill_set

log = _log.get("coach")


# --- Audio capture constants ----------------------------------------------
#
# This module is the *canonical* source for the audio-format constants below.
# They define the wire format Gemma 4's audio encoder expects (16 kHz mono
# Int16 PCM, ~50 ms frames). Every other consumer in the repo MUST match:
#   - cli/chat.py            imports SR, FRAME_MS, etc. from here
#   - web-py/backend         imports SR, MIN_SPEECH_MS via `import coach`
#   - web-py/frontend audio  hard-coded `TARGET_SAMPLE_RATE = 16000` —
#                            keep it in sync with SR below

SR = 16000
FRAME_MS = 50
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 800
SILENCE_RMS = 350.0
MIN_SPEECH_MS = 250
MAX_UTTERANCE_MS = 28_000
PREROLL_MS = 200
POST_TTS_PAUSE_MS = 350

# Coach mode allows longer pauses than open voice chat: patients may pause
# mid-utterance during phrase practice, so 800 ms of silence ends the turn
# here vs 600 ms (VOICE_END_OF_TURN_MS) in chat.py's conversational mode.
END_OF_TURN_MS = 800

MAX_RETRIES_PER_DRILL = 2

SESSION_DIR = Path(
    os.environ.get(
        "VOICE_COACH_SESSION_DIR",
        str(Path.home() / ".voice-coach" / "sessions"),
    )
)


# --- Prompt + JSON contract ----------------------------------------------

_STAGE_LABEL = {
    "warmup": "warm-up",
    "glide": "pitch glide",
    "counting": "counting",
    "main_task": "main task",
}


# --- Two-tier prompt structure (KV-cache-friendly) ------------------------
#
# The old prompt was ~3 000 tokens and was sent on EVERY drill turn — full
# honesty preamble, four worked JSON examples, and the per-drill metadata
# all in one giant system message. At ~350 prefill tps that was ~8 s of
# pure prefill before the audio encoder even ran.
#
# We split it now:
#
#   COACH_SESSION_PREAMBLE  — a short, fixed role + JSON contract sent
#                             ONCE at the start of the session and held
#                             in Cactus's KV cache for the rest of the
#                             session. Compact field names (h/m/a/f/n)
#                             cut both prompt and reply token counts.
#
#   build_drill_prompt()   — a tiny per-turn block (~80 tokens) carrying
#                             only the drill text + the live measurements.
#                             This is the only thing prefill has to chew
#                             on between turns.
#
# The praise-scrubbing safety net (`_enforce_strict_matching`) runs
# server-side regardless of what the model emits, so we don't need
# worked examples in the prompt to get the same correctness — the
# regex post-processor enforces "no praise on mismatch" deterministically.

COACH_SESSION_PREAMBLE = """You are an honest, evidence-based speech coach for adults with motor
speech disorders. Be brief and direct. Never praise a mismatch.

For every drill the user will send a short audio attempt. Reply with ONE
JSON object and nothing else (no prose, no markdown fences). Schema with
COMPACT keys:

{"h":"<heard, 1-10 words>","m":<0|1>,"a":"<ack, 3-7 words>","f":"<feedback, 8-22 words>","n":"retry"|"advance"|"rest"}

Field meanings (these long names are the canonical schema; reply with the
COMPACT keys above to save tokens):
  h = heard           — what you actually heard, transcribed verbatim.
                        For silence/cough/breath only, use "(nothing clear)".
  m = matched_prompt  — 1 if the audio matches the expected utterance,
                        0 otherwise. Bar is HIGH; when in doubt, m=0.
                        Single words/names: target word must be recognisable.
                        Sentences: key content words must be present.
                        Sustained vowels/glides: a clear vowel attempt counts.
                        Silence or only filler ("um"/"uh") is ALWAYS m=0.
  a = ack             — short acknowledgement.
                        m=0 → neutral, name what you heard, no praise.
                        m=1 → a brief warm line ("Strong voice.").
  f = feedback        — one specific, actionable cue.
                        m=0 → restate the prompt and ask them to try it.
  n = next_action     — "retry" if m=0 OR a specific element needs work;
                        "advance" if m=1 and the attempt was solid;
                        "rest" only if the patient asks for a break.
                        m=0 ALWAYS implies n="retry"."""


COACH_DRILL_TEMPLATE = """Drill {stage_label}: "{prompt}". Exercise: "{exercise_name}".
Loudness heard {achieved_dbfs:.1f} dBFS, target {target_dbfs:.1f} dBFS, duration {duration_s:.1f}s. Reply with the JSON object."""


def build_drill_prompt(
    drill: Drill, achieved_dbfs: float, duration_s: float
) -> str:
    """Per-turn delta. Tiny — designed to NOT trash the KV cache."""
    stage_label = _STAGE_LABEL.get(drill.stage, drill.stage.replace("_", " "))
    exercise_name = drill.exercise_name or "speech practice"
    return COACH_DRILL_TEMPLATE.format(
        stage_label=stage_label,
        prompt=drill.prompt,
        exercise_name=exercise_name,
        achieved_dbfs=achieved_dbfs if math.isfinite(achieved_dbfs) else -90.0,
        target_dbfs=drill.target_dbfs,
        duration_s=duration_s,
    )


# Back-compat alias: the old call site builds a single, self-contained
# system prompt that pastes preamble + delta together. New WS path uses
# the split helpers directly to win the KV-cache benefit. The CLI keeps
# calling build_prompt() until #6 wires it through, and tests pin both
# field names appearing in the rendered text.
def build_prompt(drill: Drill, achieved_dbfs: float, duration_s: float) -> str:
    """Render a self-contained prompt (preamble + per-drill delta).

    Kept as a single string so existing CLI/test call sites stay working.
    The KV-cache win comes from the WS server splitting these and pinning
    the preamble in cache via cactus_prefill (see web-py/backend).
    """
    return (
        COACH_SESSION_PREAMBLE
        + "\n\n"
        + build_drill_prompt(drill, achieved_dbfs, duration_s)
    )


_CACTUS_ENVELOPE_KEYS = {"success", "response"}


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced {...} in `text` and json.loads it.

    Tolerates markdown fences and small amounts of leading/trailing prose.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n\t")
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError as exc:
                    log.warning("JSON parse failed: %s; blob=%r", exc, blob[:200])
                    return None
    return None


# Compact ↔ canonical field-name mapping.
#
# We ask Gemma to emit compact keys (h/m/a/f/n) to cut decode tokens
# ~40 %. The rest of the code (validator, enforcer, WS payload, session
# log) is built around the canonical long names, so we normalize at the
# parse boundary. Tolerant of either shape — the CLI test suite still
# passes raw long-name dicts.
_COMPACT_TO_CANONICAL = {
    "h": "heard",
    "a": "ack",
    "f": "feedback",
    "n": "next_action",
}


def _normalize_compact_keys(obj: dict[str, Any]) -> dict[str, Any]:
    """Map h/m/a/f/n into heard/ack/feedback/next_action/metrics_observed.

    Long-name fields already on the object always win — we never overwrite
    a real value with a compact alias. `m` becomes
    `metrics_observed.matched_prompt` so the existing strict-matching
    enforcer keeps working unchanged.
    """
    if not isinstance(obj, dict):
        return obj
    for compact, canonical in _COMPACT_TO_CANONICAL.items():
        if compact in obj and canonical not in obj:
            obj[canonical] = obj.pop(compact)
        elif compact in obj:
            obj.pop(compact, None)
    if "m" in obj:
        m_raw = obj.pop("m")
        try:
            matched = bool(int(m_raw))
        except (TypeError, ValueError):
            matched = bool(m_raw)
        mo = obj.setdefault("metrics_observed", {})
        if isinstance(mo, dict):
            mo.setdefault("matched_prompt", matched)
        else:
            obj["metrics_observed"] = {"matched_prompt": matched}
    return obj


def parse_coach_json(raw: str) -> dict[str, Any] | None:
    """Extract the coach JSON from a model reply.

    Handles three wire formats:
      1. Bare JSON object with canonical long field names.
      2. Bare JSON object with compact field names (h/m/a/f/n) — the
         shape the slim COACH_SESSION_PREAMBLE asks the model to emit.
      3. Cactus envelope: {"success": .., "response": "<inner JSON string>"}
         — what `cactus_complete` writes into its output buffer when the
         caller reads the buffer directly instead of streaming tokens.

    Returns None if none yields a usable object. Compact keys are
    normalised to the canonical names at the boundary so the validator,
    enforcer, WS payload and session log all keep working unchanged.
    """
    obj = _extract_first_json_object(raw)
    if obj is None:
        return None
    # Unwrap Cactus envelope if present and our schema keys are not at top level.
    if (
        isinstance(obj, dict)
        and _CACTUS_ENVELOPE_KEYS.issubset(obj)
        and "ack" not in obj
        and "a" not in obj
        and isinstance(obj.get("response"), str)
    ):
        log.debug("unwrapping Cactus envelope")
        inner = _extract_first_json_object(obj["response"])
        if inner is not None:
            obj = inner
    if isinstance(obj, dict):
        obj = _normalize_compact_keys(obj)
    return obj


# Words/phrases the model uses to praise — if any of these slip into
# `ack` while matched_prompt is false, the enforcer rewrites the line.
# Kept short and HIGH-PRECISION on purpose: false positives here would
# wipe out legitimate "Strong voice" / "Clear and full" feedback when
# the prompt actually matched.
_PRAISE_TOKENS = (
    "nice", "great", "good job", "good attempt", "good try",
    "excellent", "well done", "wonderful", "fantastic", "perfect",
    "awesome", "amazing", "beautiful", "lovely", "bravo",
    "way to go", "keep it up", "you got it", "love that",
)
_PRAISE_RE = None  # built lazily from _PRAISE_TOKENS


def _looks_like_praise(text: str) -> bool:
    """Case-insensitive substring check against the praise vocabulary."""
    global _PRAISE_RE
    if _PRAISE_RE is None:
        import re
        _PRAISE_RE = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in _PRAISE_TOKENS) + r")\b",
            re.IGNORECASE,
        )
    return bool(_PRAISE_RE.search(text or ""))


def _enforce_strict_matching(obj: dict[str, Any]) -> dict[str, Any]:
    """Override inconsistent fields so the coach never praises a mismatch.

    Even with worked examples in the system prompt, Gemma 4 sometimes
    returns matched_prompt=false with next_action=advance and a
    cheerful "nice try!" ack. That's the exact failure mode that makes
    the coach untrustworthy ("the app told me 'good job' even though I
    said something completely different"). Rather than relying on the
    model's discipline alone, the server enforces:

      matched_prompt = false  ⇒  next_action = "retry"
      matched_prompt = false  ⇒  ack contains no praise vocabulary
      matched_prompt = false  ⇒  feedback names the original prompt

    Mutates `obj` in place and returns it. A no-op when matched_prompt
    is true or absent — this intentionally never overrides the model's
    positive judgments.
    """
    mo = obj.get("metrics_observed") or {}
    matched = mo.get("matched_prompt", True)
    if matched is not False:
        return obj

    original_action = obj.get("next_action")
    if original_action != "retry":
        log.info(
            "strict-match: model returned next_action=%r with "
            "matched_prompt=false — forcing retry",
            original_action,
        )
        obj["next_action"] = "retry"

    ack = (obj.get("ack") or "").strip()
    heard = (obj.get("heard") or "").strip()
    if _looks_like_praise(ack):
        log.info(
            "strict-match: scrubbing praise from ack=%r (matched_prompt=false)",
            ack,
        )
        # Replace with a neutral, honest acknowledgement that NAMES the
        # mismatch. Keeps the user's trust in the coach calibrated.
        if heard and heard != "(nothing clear)":
            obj["ack"] = f"I heard '{heard}'."
        elif heard == "(nothing clear)":
            obj["ack"] = "I didn't catch that."
        else:
            obj["ack"] = "Not quite."

    return obj


def validate_coach_json(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not obj:
        return None
    required = {"ack", "feedback", "next_action"}
    missing = required - set(obj)
    if missing:
        log.warning("coach JSON missing keys: %s", sorted(missing))
        return None
    if obj["next_action"] not in ("retry", "advance", "rest"):
        log.warning("coach JSON invalid next_action: %r", obj["next_action"])
        return None
    obj.setdefault("metrics_observed", {})
    # Strict-matching enforcement. Runs AFTER schema validation so we
    # only ever rewrite well-formed payloads.
    obj = _enforce_strict_matching(obj)
    return obj


def rms_to_dbfs(rms: float) -> float:
    if rms <= 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


# --- Mic capture ----------------------------------------------------------

def capture_one_utterance(np_mod, sd_mod) -> tuple[bytes, float, float]:  # pragma: no cover - needs sounddevice + microphone
    """Block until one utterance is captured.

    Returns (pcm_bytes, avg_dbfs_over_voiced_frames, duration_s).
    Returns (b"", -inf, 0.0) if nothing was heard.
    """
    end_silence = max(1, END_OF_TURN_MS // FRAME_MS)
    min_speech = max(1, MIN_SPEECH_MS // FRAME_MS)
    max_frames = MAX_UTTERANCE_MS // FRAME_MS
    preroll_n = max(0, PREROLL_MS // FRAME_MS)
    post_tts_pause = POST_TTS_PAUSE_MS / 1000.0

    speech_started = False
    speech_frames = 0
    silence_frames = 0
    total_frames = 0
    captured: list[bytes] = []
    preroll: list[bytes] = []
    voiced_rms: list[float] = []

    try:
        stream_ctx = sd_mod.InputStream(
            samplerate=SR, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
        )
    except Exception as exc:
        die(
            ExitCode.ENV_NO_AUDIO_DEVICE,
            f"Could not open microphone: {exc}",
            "Check System Settings -> Privacy & Security -> Microphone.",
            "List devices: python3.14 -c 'import sounddevice as sd; print(sd.query_devices())'",
        )

    with stream_ctx as mic:
        # Drain frames buffered while TTS was playing -- those are echoes
        # of our own voice through the speakers.
        drain_until = time.monotonic() + post_tts_pause
        while time.monotonic() < drain_until:
            try:
                mic.read(FRAME_SAMPLES)
            except Exception:
                break

        while True:
            try:
                data, _ = mic.read(FRAME_SAMPLES)
            except Exception as exc:
                die(
                    ExitCode.RUNTIME_AUDIO_FAILED,
                    f"Mic read failed mid-utterance: {exc}",
                )
            frame = np_mod.asarray(data, dtype=np_mod.int16).reshape(-1)
            if frame.size == 0:
                continue
            rms = float(
                np_mod.sqrt(np_mod.mean(frame.astype(np_mod.float32) ** 2))
            )
            is_speech = rms > SILENCE_RMS
            raw = frame.tobytes()

            if is_speech:
                if not speech_started:
                    speech_started = True
                    captured.extend(preroll)
                    preroll.clear()
                speech_frames += 1
                silence_frames = 0
                voiced_rms.append(rms)
            elif speech_started:
                silence_frames += 1
            else:
                preroll.append(raw)
                if len(preroll) > preroll_n:
                    preroll.pop(0)

            if speech_started:
                captured.append(raw)
                total_frames += 1

            if (
                speech_started
                and speech_frames >= min_speech
                and silence_frames >= end_silence
            ):
                break
            if total_frames >= max_frames:
                log.debug("max utterance length hit (%d frames)", total_frames)
                break

    if not captured or speech_frames < min_speech:
        return b"", -math.inf, 0.0

    pcm = b"".join(captured)
    duration = len(pcm) / 2 / SR
    avg_rms = sum(voiced_rms) / len(voiced_rms) if voiced_rms else 0.0
    return pcm, rms_to_dbfs(avg_rms), duration


# --- Session log ----------------------------------------------------------

def _open_session_log() -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SESSION_DIR / f"session-{ts}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("failed to write session log %s: %s", path, exc)


# --- Coach loop -----------------------------------------------------------

_TRAILING_PUNCT = " \t\n\r.!?,;:"


def _join_for_speech(ack: str, feedback: str) -> str:
    """Combine ack + feedback into one TTS line with no double punctuation.

    Empty / whitespace-only inputs are skipped entirely so we never emit a
    bare ". ." utterance to the speech engine.
    """
    parts: list[str] = []
    for piece in (ack, feedback):
        cleaned = piece.strip().rstrip(_TRAILING_PUNCT)
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return ""
    return ". ".join(parts) + "."


class _TokenCollector:
    """Accumulate streamed tokens into a single string.

    `cactus_complete` writes a Cactus-envelope JSON into its output buffer
    ({"success", "response": "<inner>"}) but streams ONLY the inner response
    text through the per-token callback. Collecting via callback gives us
    the model's reply directly with no envelope to unwrap.
    """

    __slots__ = ("parts",)

    def __init__(self) -> None:
        self.parts: list[str] = []

    def __call__(self, text: str, _tid: int) -> None:
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "".join(self.parts)


def coach_mode(  # pragma: no cover - full hardware+model integration; covered by test_server.py via WebSocket
    cactus_module,                       # imported `cactus` module
    cactus_complete_audio: Callable,     # _cactus_complete_audio from chat.py
    speak_blocking: Callable[[str, str | None], None],
    stop_speaking: Callable[[], None],
    weights_path: Path,
    voice_name: str | None,
    temperature: float = 0.4,
    max_tokens: int = 256,
) -> None:
    """Run the drill-driven coach loop until done, rest, or Ctrl+C."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        die(
            ExitCode.ENV_MISSING_PYTHON_DEP,
            "Coach mode requires sounddevice + numpy.",
            "Install: python3.14 -m pip install --break-system-packages sounddevice numpy",
            "Or use ./run-cli which installs them automatically.",
        )

    log.info("loading model from %s", weights_path)
    try:
        model = cactus_module.cactus_init(str(weights_path), None, False)
    except Exception as exc:
        die(
            ExitCode.SETUP_MODEL_LOAD_FAILED,
            f"cactus_init failed: {exc}",
            "Re-download with: cactus download <model> --reconvert",
        )

    drills = default_drill_set()
    session_path = _open_session_log()
    log.info("session log: %s", session_path)

    options = json.dumps({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "confidence_threshold": 0.0,
    })

    # Reset ONCE at session start so the model has a clean cache. We do
    # NOT reset per turn — keeping the cache means the COACH_SESSION_PREAMBLE
    # at messages[0] stays prefilled (chunked prefill prefix-matches the
    # identical preamble each turn) and only the small per-drill delta +
    # the new audio frames are encoded. This is the bulk of the latency
    # win from the cactus-style optimization.
    try:
        cactus_module.cactus_reset(model)
    except Exception as exc:
        log.debug("session-start cactus_reset raised %s (continuing)", exc)

    interrupted = False

    def handle_sigint(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        stop_speaking()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    advanced = 0
    retries_total = 0
    rest_called = False
    json_failures = 0
    dbfs_seen: list[float] = []

    # cactus-transcribe-style startup line: one row, the way `cactus
    # transcribe` and `cactus run` introduce themselves. Two flag chips
    # advertise the optimizations the user is now benefiting from so a
    # casual reader of the terminal sees they're live without diffing.
    print()
    print(
        f"▸ Voice Coach  [{len(drills)} drills]  "
        "kv-cache: pinned  ·  prompt: slim  ·  tts: streamed  ·  Ctrl+C to stop"
    )

    try:
        i = 0
        retries_for_drill = 0
        while i < len(drills) and not interrupted and not rest_called:
            drill = drills[i]
            # One-line drill banner: stage, position, expected utterance.
            # Avoids the multi-line "label / note / listening" preamble the
            # old loop used — single status line per turn keeps the
            # terminal scrollback as terse as `cactus transcribe`.
            print(f'\n▸ [{drill.stage} {drill.index + 1}/{len(drills)}] "{drill.prompt}"')
            speak_blocking(f"Please say: {drill.prompt}", voice_name)

            pcm, dbfs, duration = capture_one_utterance(np, sd)
            if not pcm:
                print("   ⏺ (no speech detected)")
                retries_for_drill += 1
                if retries_for_drill > MAX_RETRIES_PER_DRILL:
                    log.info(
                        "skipping drill %s/%d after %d empty captures",
                        drill.stage, drill.index, retries_for_drill,
                    )
                    i += 1
                    retries_for_drill = 0
                continue

            dbfs_seen.append(dbfs)

            # KV-cache-friendly layout: preamble lives at index 0
            # byte-identically every turn, so cactus's prefix match keeps
            # it pinned in the cache. Only the tiny per-drill delta has
            # to be re-prefilled — and the audio encoder runs over the
            # new PCM. NO cactus_reset between turns.
            messages = [
                {"role": "system", "content": COACH_SESSION_PREAMBLE},
                {"role": "system", "content": build_drill_prompt(drill, dbfs, duration)},
                {"role": "user", "content": ""},
            ]

            log.debug(
                "drill=%s/%d sending %d audio bytes (%.2fs)",
                drill.stage, drill.index, len(pcm), duration,
            )
            collector = _TokenCollector()
            t0 = time.monotonic()
            try:
                envelope = cactus_complete_audio(
                    cactus_module, model,
                    json.dumps(messages),
                    options,
                    pcm,
                    collector,
                )
            except Exception as exc:
                log.error("model call failed on %s/%d: %s", drill.stage, drill.index, exc)
                _append_jsonl(session_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drill": asdict(drill),
                    "achieved_dbfs": dbfs,
                    "duration_s": duration,
                    "error": str(exc),
                })
                retries_for_drill += 1
                if retries_for_drill > MAX_RETRIES_PER_DRILL:
                    i += 1
                    retries_for_drill = 0
                continue

            latency = time.monotonic() - t0
            streamed = collector.text()
            # Prefer the streamed callback text (it's just the model reply);
            # fall back to the envelope buffer if streaming was empty.
            raw_for_parse = streamed or envelope
            log.debug(
                "model latency %.2fs streamed=%dB envelope=%dB sample=%r",
                latency, len(streamed), len(envelope), raw_for_parse[:200],
            )

            valid = validate_coach_json(parse_coach_json(raw_for_parse))
            if not valid:
                json_failures += 1
                print("   (model produced invalid JSON — moving on)")
                _append_jsonl(session_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drill": asdict(drill),
                    "achieved_dbfs": dbfs,
                    "duration_s": duration,
                    "model_latency_s": round(latency, 3),
                    "streamed_reply": streamed,
                    "envelope_reply": envelope,
                    "parse_error": True,
                })
                i += 1
                retries_for_drill = 0
                continue

            heard = (valid.get("heard") or "").strip()
            ack = (valid.get("ack") or "").strip()
            feedback = (valid.get("feedback") or "").strip()
            action = valid["next_action"]
            matched = bool(valid.get("metrics_observed", {}).get("matched_prompt", True))

            # Single status line per turn, mirroring the compact look
            # of `cactus transcribe` output. Wall-clock latency is the
            # main demo-time signal so we render it inline.
            mark = "✓" if matched else "✗"
            heard_chip = f' "{heard}"' if heard else ""
            print(
                f"   ⏺ {duration:.1f}s {dbfs:.1f}dBFS  "
                f"→{heard_chip}  {mark} {action}  ({latency:.1f}s)"
            )
            spoken = _join_for_speech(ack, feedback)
            if spoken:
                print(f"   ◂ {spoken}")
                # Sentence-stream into `say` so playback starts on
                # sentence 1 immediately, instead of waiting for the
                # whole reply line to render. Mirrors the WS path
                # which ships one audio_reply chunk per sentence.
                done, leftover = split_sentences(spoken)
                pieces = list(done)
                leftover = leftover.strip()
                if leftover:
                    pieces.append(leftover)
                for piece in pieces:
                    speak_blocking(piece, voice_name)

            _append_jsonl(session_path, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "drill": asdict(drill),
                "achieved_dbfs": dbfs,
                "duration_s": duration,
                "model_latency_s": round(latency, 3),
                "coach": valid,
            })

            if action == "rest":
                rest_called = True
                print("   (coach suggested rest — wrapping up)")
                break
            if action == "retry" and retries_for_drill < MAX_RETRIES_PER_DRILL:
                retries_for_drill += 1
                retries_total += 1
                continue

            advanced += 1
            i += 1
            retries_for_drill = 0

    except KeyboardInterrupt:
        print("\n(stopping)")
    finally:
        stop_speaking()
        try:
            cactus_module.cactus_destroy(model)
        except Exception as exc:
            log.debug("cactus_destroy raised %s", exc)

    print()
    print("─── session summary ───")
    print(f"  drills completed: {advanced}/{len(drills)}")
    print(f"  retries:          {retries_total}")
    if dbfs_seen:
        avg = sum(dbfs_seen) / len(dbfs_seen)
        print(f"  avg loudness:     {avg:.1f} dBFS")
    if json_failures:
        print(f"  JSON failures:    {json_failures}")
    print(f"  session log:      {session_path}")
    if rest_called:
        print("  ended early (rest requested)")
