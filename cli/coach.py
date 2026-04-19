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
from content import Drill, default_drill_set

log = _log.get("coach")


# --- Audio capture constants (kept in sync with chat.py voice mode) -------

SR = 16000
FRAME_MS = 50
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 800
SILENCE_RMS = 350.0
END_OF_TURN_MS = 800
MIN_SPEECH_MS = 250
MAX_UTTERANCE_MS = 28_000
PREROLL_MS = 200
POST_TTS_PAUSE_MS = 350

MAX_RETRIES_PER_DRILL = 2

SESSION_DIR = Path(
    os.environ.get(
        "VOICE_COACH_SESSION_DIR",
        str(Path.home() / ".voice-coach" / "sessions"),
    )
)


# --- Prompt + JSON contract ----------------------------------------------

COACH_SYSTEM_TEMPLATE = """You are a warm, patient speech coach for adults with motor speech
disorders (Parkinson's, post-stroke dysarthria).

The patient is on a {stage} drill. They were asked to say:
"{prompt}"

Measured signal (relative, NOT calibrated to room SPL):
- average voiced loudness: {achieved_dbfs:.1f} dBFS
- target loudness:         {target_dbfs:.1f} dBFS
- utterance duration:      {duration_s:.1f} s

You will hear their attempt as audio. Listen for: vocal effort, breath
support, pitch monotony, trailing off, rushed pace, slurred articulation.
Be specific but never discouraging.

Respond with JSON ONLY. No prose, no markdown fences. Schema:
{{
  "ack": "<one short warm acknowledgement, 3-6 words>",
  "feedback": "<one specific, actionable cue, 8-18 words>",
  "next_action": "retry" | "advance" | "rest",
  "metrics_observed": {{
    "loudness_ok": true | false,
    "pitch_range_ok": true | false,
    "pace_ok": true | false,
    "articulation_ok": true | false
  }}
}}"""


def build_prompt(drill: Drill, achieved_dbfs: float, duration_s: float) -> str:
    return COACH_SYSTEM_TEMPLATE.format(
        stage=drill.stage,
        prompt=drill.prompt,
        achieved_dbfs=achieved_dbfs if math.isfinite(achieved_dbfs) else -90.0,
        target_dbfs=drill.target_dbfs,
        duration_s=duration_s,
    )


def parse_coach_json(raw: str) -> dict[str, Any] | None:
    """Extract the first balanced {...} object from `raw`.

    Tolerates markdown fences and small amounts of leading/trailing prose.
    Returns None if no valid JSON object is found.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n\t")
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
    return obj


def rms_to_dbfs(rms: float) -> float:
    if rms <= 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)


# --- Mic capture ----------------------------------------------------------

def capture_one_utterance(np_mod, sd_mod) -> tuple[bytes, float, float]:
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

def _noop_token(_text: str, _tid: int) -> None:
    """Discard streamed tokens — coach mode wants the full JSON, not a stream."""
    pass


def coach_mode(
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

    print()
    print("Voice coach session starting. Speak when prompted. Ctrl+C ends early.")
    print()

    try:
        i = 0
        retries_for_drill = 0
        while i < len(drills) and not interrupted and not rest_called:
            drill = drills[i]
            label = f"[{drill.stage} {drill.index + 1}]"
            print(f"\n{label} say: \"{drill.prompt}\"")
            if drill.note:
                print(f"   {drill.note}")
            speak_blocking(f"Please say: {drill.prompt}", voice_name)

            print("listening…")
            pcm, dbfs, duration = capture_one_utterance(np, sd)
            if not pcm:
                print("   (no speech detected)")
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
            print(
                f"   captured {duration:.1f}s, avg loudness {dbfs:.1f} dBFS"
            )

            sys_prompt = build_prompt(drill, dbfs, duration)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ""},
            ]

            try:
                cactus_module.cactus_reset(model)
            except Exception as exc:
                log.debug("cactus_reset raised %s (continuing)", exc)

            log.debug(
                "drill=%s/%d sending %d audio bytes (%.2fs)",
                drill.stage, drill.index, len(pcm), duration,
            )
            t0 = time.monotonic()
            try:
                raw = cactus_complete_audio(
                    cactus_module, model,
                    json.dumps(messages),
                    options,
                    pcm,
                    _noop_token,
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
            log.debug("model latency %.2fs raw=%r", latency, raw[:200])

            valid = validate_coach_json(parse_coach_json(raw))
            if not valid:
                json_failures += 1
                print("   (model produced invalid JSON — moving on)")
                _append_jsonl(session_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drill": asdict(drill),
                    "achieved_dbfs": dbfs,
                    "duration_s": duration,
                    "model_latency_s": round(latency, 3),
                    "raw_reply": raw,
                    "parse_error": True,
                })
                i += 1
                retries_for_drill = 0
                continue

            ack = (valid.get("ack") or "").strip()
            feedback = (valid.get("feedback") or "").strip()
            action = valid["next_action"]

            spoken = ". ".join(part for part in (ack, feedback) if part)
            print(f"   coach: {spoken}")
            if spoken:
                speak_blocking(spoken, voice_name)

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
