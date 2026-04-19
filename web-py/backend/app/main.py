"""FastAPI server for the Voice Coach web app.

Single WebSocket endpoint `/ws/coach` drives the drill loop. The browser
handles mic capture (16 kHz mono Int16 PCM, base64-encoded over the wire);
the server runs three on-device models via Cactus on the user's machine:

  - Gemma 4 E2B (~4 GB, INT4)     audio-native multimodal coaching
  - Whisper-tiny (~80 MB, INT4)   voice-command transcription
  - FunctionGemma 270M (~150 MB)  intent → action routing

Server-side TTS uses macOS `say` so the coach's reply audio is also
generated locally (no browser/cloud TTS needed). The frontend falls
back to `window.speechSynthesis` if no WAV arrives.

We reuse cli/ wholesale:
  - cli/_log.py            structured logging
  - cli/_exit.py           typed exit codes (used as error codes over WS)
  - cli/content.py         drill content set (Drill dataclass, default_drill_set)
  - cli/coach.py           prompt builder, JSON parser/validator, dBFS helper,
                           session-log helpers, _TokenCollector
  - cli/chat.py            ensure_lib_discoverable, ensure_model,
                           _cactus_complete_audio (the FFI helper)
  - cli/intent.py          Intent enum + classifiers + classify()

Wire format
-----------
Client -> Server:
  { "type": "start_session" }
  { "type": "audio", "pcm_b64": "<base64 int16 PCM>", "sample_rate": 16000 }
  { "type": "command", "action": "skip" | "rest" | "repeat_prompt" }
  { "type": "intent", "utterance": "<free-form text>" }
  { "type": "intent_audio", "pcm_b64": "<base64 int16 PCM>", "sample_rate": 16000 }

Server -> Client:
  { "type": "loading" }
  { "type": "ready" }
  { "type": "drill", stage, index, prompt, note, target_dbfs, total, position }
  { "type": "metrics", dbfs, duration_s }
  { "type": "thinking", step, label }
       # `step` is one of:
       #   "analyzing_audio"      — PCM accepted, Gemma is consuming the audio
       #   "generating_response"  — first response token streamed back
       #   "parsing_response"     — model call returned, parsing JSON envelope
       #   "synthesizing_voice"   — rendering the spoken reply via macOS `say`
       # `label` is a short human-readable phrase the UI can show as-is.
       # The backend sends one frame per step, in order, so the client can
       # stack them into a live progress list (with per-step elapsed time)
       # instead of a static "Coach is thinking…" placeholder.
  { "type": "coach", heard, matched_prompt, ack, feedback,
                     next_action, metrics_observed, latency_s }
  { "type": "audio_reply", wav_b64, source }    # macOS `say` rendered TTS
  { "type": "advance" | "retry" | "rest" }
  { "type": "intent_result", action, confidence, utterance, source, latency_ms,
                              transcribe_latency_ms, transcript }
  { "type": "session_done", summary }
  { "type": "error", code, message }
"""
from __future__ import annotations

import array
import asyncio
import base64
import json
import math
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Put cli/ on sys.path BEFORE importing the cli modules.
REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_DIR = REPO_ROOT / "cli"
if str(CLI_DIR) not in sys.path:  # pragma: no cover - depends on import order at process start
    sys.path.insert(0, str(CLI_DIR))

import _log                            # noqa: E402  (cli module on sys.path)
from _exit import ExitCode             # noqa: E402
import chat                            # noqa: E402
import coach                           # noqa: E402
import content                         # noqa: E402
import intent as intent_mod            # noqa: E402  (avoid shadowing local `intent` vars)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware           # noqa: E402

log = _log.get("server")

DEFAULT_MODEL_ID = "google/gemma-4-e2b-it"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 256

# FunctionGemma 270M is a separate, smaller model used purely for routing
# free-form patient utterances to in-app actions (skip / rest / repeat).
# It loads independently of Gemma 4 so the coach is never blocked on it,
# and the WS handler degrades cleanly to a regex heuristic when it is
# unavailable. Override via VOICE_COACH_FUNCGEMMA_ID for experiments.
import os as _os                                         # noqa: E402
import shutil                                            # noqa: E402
import subprocess                                        # noqa: E402
import tempfile                                          # noqa: E402

INTENT_MODEL_ID = _os.environ.get(
    "VOICE_COACH_FUNCGEMMA_ID", intent_mod.DEFAULT_FUNCGEMMA_ID,
)

# Whisper-tiny is the smallest STT model in the Cactus catalogue: 39M
# params, ~230 ms end-to-end on Mac M3 per the Cactus benchmarks.
# We use it ONLY for voice-command transcription (typically <2 s of
# audio). Drill audio still goes straight to Gemma 4 — its audio
# encoder is what gives the coach prosodic awareness; transcribing
# first would throw that away.
WHISPER_MODEL_ID = _os.environ.get(
    "VOICE_COACH_WHISPER_ID", "openai/whisper-tiny",
)

# macOS `say` voice for server-side TTS. None = system default.
# `--data-format=LEI16@22050` makes `say` write a real PCM WAV that
# every browser plays without fuss.
SAY_VOICE = _os.environ.get("VOICE_COACH_SAY_VOICE") or None
SAY_DATA_FORMAT = "LEI16@22050"
# If `say` isn't present (Linux/CI/Docker), TTS WAV is simply skipped
# and the client falls back to its own `speechSynthesis`. Detected once
# at import time so we don't shell out per request.
HAS_SAY = shutil.which("say") is not None


class ModelHolder:
    """Lazily loads the Cactus model once and reuses it across sessions."""

    def __init__(self) -> None:
        self.cactus = None              # the cactus python module
        self.model = None               # opaque model handle
        self.weights_path: Path | None = None
        self.load_error: str | None = None
        self.options_json: str = json.dumps({
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "confidence_threshold": 0.0,
        })
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    async def ensure_loaded(self) -> None:  # pragma: no cover - real load needs Cactus dylib
        if self.loaded or self.load_error:
            return
        async with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                # Heavy work runs in a thread so it doesn't block the loop.
                await asyncio.to_thread(self._load_blocking)
            except Exception as exc:
                log.exception("model load failed")
                self.load_error = str(exc)

    def _load_blocking(self) -> None:  # pragma: no cover - calls into native lib + downloads weights
        chat.ensure_lib_discoverable()
        self.weights_path = chat.ensure_model(DEFAULT_MODEL_ID)
        import cactus  # type: ignore  # late import: needs dylib + bindings
        self.cactus = cactus
        log.info("loading model from %s", self.weights_path)
        self.model = cactus.cactus_init(str(self.weights_path), None, False)
        log.info("model ready")


class IntentHolder:
    """Lazily loads FunctionGemma 270M and exposes a classifier.

    Independent of `ModelHolder` (Gemma 4 chat). The two models load in
    parallel; either may finish first. While FunctionGemma is loading
    (or if its load fails), `classifier()` returns None and callers
    fall back to the regex heuristic in cli/intent.py — the demo never
    blocks waiting for the router.
    """

    def __init__(self) -> None:
        self.cactus = None
        self.model = None
        self.weights_path: Path | None = None
        self.load_error: str | None = None
        self._classifier: intent_mod.FunctionGemmaClassifier | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._classifier is not None

    def classifier(self) -> intent_mod.FunctionGemmaClassifier | None:
        return self._classifier

    async def ensure_loaded(self) -> None:  # pragma: no cover - real load needs Cactus dylib
        if self.loaded or self.load_error:
            return
        async with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                await asyncio.to_thread(self._load_blocking)
            except Exception as exc:
                # Soft-failure: log and keep serving with the heuristic
                # fallback. We never want a bad FunctionGemma install to
                # take down the coach app.
                log.warning("FunctionGemma load failed (%s) — using heuristic", exc)
                self.load_error = str(exc)

    def _load_blocking(self) -> None:  # pragma: no cover - calls into native lib
        chat.ensure_lib_discoverable()
        self.weights_path = chat.ensure_model(INTENT_MODEL_ID)
        import cactus  # type: ignore  # late import: needs dylib + bindings
        self.cactus = cactus
        log.info("loading intent model from %s", self.weights_path)
        self.model = cactus.cactus_init(str(self.weights_path), None, False)

        # Wrap the FFI call in a thread-safe completion closure that
        # FunctionGemmaClassifier can call without knowing about Cactus.
        # We use the streaming token collector pattern from coach mode
        # so we get the model's reply directly (envelope-free).
        def _complete(messages_json: str, options_json: str) -> str:
            try:
                cactus.cactus_reset(self.model)
            except Exception as exc:  # pragma: no cover - defensive; reset failures are non-fatal
                log.debug("intent cactus_reset raised %s", exc)
            collector = coach._TokenCollector()
            envelope = chat._cactus_complete_audio(
                cactus, self.model,
                messages_json,
                options_json,
                b"",                # text-only call: no PCM
                collector,
            )
            return collector.text() or envelope

        self._classifier = intent_mod.FunctionGemmaClassifier(_complete)
        log.info("intent router ready (%s)", INTENT_MODEL_ID)


class WhisperHolder:
    """Lazily loads Cactus Whisper for voice-command transcription.

    Independent of the coach (Gemma 4) and the router (FunctionGemma);
    all three load in parallel. While Whisper is loading or if its load
    fails, the WS handler refuses `intent_audio` requests and the
    client is expected to fall back to the typed-text path so the
    demo never blocks on an in-progress download.
    """

    def __init__(self) -> None:
        self.cactus = None
        self.model = None
        self.weights_path: Path | None = None
        self.load_error: str | None = None
        self.options_json: str = json.dumps({
            "temperature": 0.0,
            "max_tokens": 64,
            "confidence_threshold": 0.0,
        })
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    async def ensure_loaded(self) -> None:  # pragma: no cover - real load needs Cactus dylib
        if self.loaded or self.load_error:
            return
        async with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                await asyncio.to_thread(self._load_blocking)
            except Exception as exc:
                log.warning("Whisper load failed (%s) — voice commands degraded", exc)
                self.load_error = str(exc)

    def _load_blocking(self) -> None:  # pragma: no cover - calls into native lib
        chat.ensure_lib_discoverable()
        self.weights_path = chat.ensure_model(WHISPER_MODEL_ID)
        import cactus  # type: ignore
        self.cactus = cactus
        log.info("loading whisper from %s", self.weights_path)
        self.model = cactus.cactus_init(str(self.weights_path), None, False)
        log.info("whisper ready (%s)", WHISPER_MODEL_ID)

    def transcribe(self, pcm: bytes) -> tuple[str, int]:  # pragma: no cover - calls into native lib
        """Transcribe raw 16 kHz mono int16 LE PCM to text.

        Returns (transcript, latency_ms). Latency is wall-clock around
        the FFI call only — it is what the demo chip should display.
        """
        if not self.loaded:
            raise RuntimeError("whisper not loaded")
        t0 = time.monotonic()
        # Use the Cactus python wrapper directly. For short command
        # utterances (a few hundred KB), the byte-by-byte ctypes copy
        # the wrapper does is fine; a fast-path memcpy isn't needed
        # the way it is for the much larger drill audio.
        try:
            self.cactus.cactus_reset(self.model)
        except Exception as exc:
            log.debug("whisper cactus_reset raised %s", exc)
        text = self.cactus.cactus_transcribe(
            self.model,
            None,                # audio_path: not used, we pass PCM
            "",                  # initial prompt
            self.options_json,
            None,                # token callback
            pcm,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return text.strip(), latency_ms


holder = ModelHolder()
intent_holder = IntentHolder()
whisper_holder = WhisperHolder()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Kick off all three model loads eagerly so the first WS connection
    # finds them ready. They run in parallel; the WS handler waits only
    # on the coach model (Gemma 4). The intent router and Whisper are
    # best-effort — both degrade cleanly if their load fails.
    asyncio.create_task(holder.ensure_loaded())
    asyncio.create_task(intent_holder.ensure_loaded())
    asyncio.create_task(whisper_holder.ensure_loaded())
    yield


app = FastAPI(title="Voice Coach", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": holder.loaded,
        "model_load_error": holder.load_error,
        "model_id": DEFAULT_MODEL_ID,
        # Intent router is independent and best-effort — clients can
        # show a "router ready" badge without blocking on this.
        "intent_loaded": intent_holder.loaded,
        "intent_load_error": intent_holder.load_error,
        "intent_model_id": INTENT_MODEL_ID,
        # Whisper STT is also best-effort. When unloaded, the frontend
        # voice-command mic falls back to typed input only.
        "whisper_loaded": whisper_holder.loaded,
        "whisper_load_error": whisper_holder.load_error,
        "whisper_model_id": WHISPER_MODEL_ID,
        # macOS `say` availability: drives whether the server emits
        # `audio_reply` frames or expects the client to TTS locally.
        "tts_available": HAS_SAY,
    }


@app.get("/api/drills")
async def drills() -> list[dict]:
    return [asdict(d) for d in content.default_drill_set()]


def _say_to_wav(text: str) -> bytes | None:
    """Render `text` to a PCM WAV using macOS `say`.

    Returns the WAV bytes (browser-playable) or None if `say` is not
    available, the input was empty, or the subprocess failed. Failure
    is non-fatal: callers should skip the audio frame and let the
    client fall back to its own speechSynthesis.
    """
    cleaned = (text or "").strip()
    if not cleaned or not HAS_SAY:
        return None
    # Render to a temp file we own. `say` writes a real PCM WAV when
    # given --data-format=LEI16@<rate> with a .wav output path.
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            args = ["say", "--data-format=" + SAY_DATA_FORMAT,
                    "-o", str(wav_path)]
            if SAY_VOICE:
                args += ["-v", SAY_VOICE]
            args += ["--", cleaned]
            proc = subprocess.run(
                args, check=False, capture_output=True, timeout=10,
            )
            if proc.returncode != 0:
                log.warning("say failed rc=%d stderr=%r",
                            proc.returncode, proc.stderr[:200])
                return None
            data = wav_path.read_bytes()
            if not data:
                return None
            return data
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("say render failed: %s", exc)
        return None


_TTS_TRAILING_PUNCT_RE = None  # populated lazily inside _build_drill_tts


def _build_drill_tts(prompt: str, note: str) -> str:
    """Mirror of the frontend's `buildDrillTTS` so the server-side
    `say` rendering speaks the same line the browser would.

    For instruction-only phases (warmup, glide), `note` is empty and the
    prompt IS the full instruction — speak it. For phases with explicit
    content (counting, main_task), speak the cue then the expected
    utterance.
    """
    global _TTS_TRAILING_PUNCT_RE
    if _TTS_TRAILING_PUNCT_RE is None:
        import re
        _TTS_TRAILING_PUNCT_RE = re.compile(r"[.!?,;: ]+$")
    n = (note or "").strip()
    p = (prompt or "").strip()
    if not n:
        return p
    return f"{_TTS_TRAILING_PUNCT_RE.sub('', n)}. Now: {p}."


def _pcm_bytes_to_dbfs(pcm: bytes) -> tuple[float, float]:
    """Return (dbfs, duration_s) for raw int16 little-endian PCM at 16 kHz."""
    if not pcm:
        return -math.inf, 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:  # pragma: no cover - frombytes always yields >=1 sample for non-empty pcm
        return -math.inf, 0.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    duration_s = len(samples) / coach.SR
    return coach.rms_to_dbfs(rms), duration_s


@app.websocket("/ws/coach")
async def ws_coach(ws: WebSocket) -> None:
    await ws.accept()
    log.info("client connected from %s", ws.client)

    drills_list = content.default_drill_set()
    drill_idx = 0
    retries_for_drill = 0
    advanced = 0
    retries_total = 0
    rest_called = False
    json_failures = 0
    dbfs_seen: list[float] = []
    session_path = coach._open_session_log()
    log.info("session log: %s", session_path)

    async def send(payload: dict) -> None:
        await ws.send_text(json.dumps(payload, default=str))

    async def send_current_drill() -> None:
        d = drills_list[drill_idx]
        # Render the drill prompt to a WAV via macOS `say` so the
        # client uses ONE consistent voice throughout the session
        # (otherwise drill prompts speak via browser speechSynthesis
        # and coach replies speak via `say`, alternating two
        # different voices and sounding like two engines are talking).
        # Skipped silently when `say` isn't available; client falls
        # back to its own TTS in that case.
        prompt_wav_b64: str | None = None
        if HAS_SAY:
            tts_text = _build_drill_tts(d.prompt, d.note)
            wav = await asyncio.to_thread(_say_to_wav, tts_text)
            if wav is not None:
                prompt_wav_b64 = base64.b64encode(wav).decode("ascii")
        payload = {
            "type": "drill",
            "stage": d.stage,
            "index": d.index,
            "prompt": d.prompt,
            "note": d.note,
            "target_dbfs": d.target_dbfs,
            "total": len(drills_list),
            "position": drill_idx,
            # Richer context (clinically-grounded program model):
            "category_id": d.category_id,
            "category_name": d.category_name,
            "exercise_id": d.exercise_id,
            "exercise_name": d.exercise_name,
            "focus": d.focus,
            "target_repetitions": d.target_repetitions,
            "target_duration_sec": d.target_duration_sec,
        }
        if prompt_wav_b64 is not None:
            payload["prompt_wav_b64"] = prompt_wav_b64
        await send(payload)

    # Snapshot of capabilities so the client can decide deterministically
    # whether to use its own speechSynthesis. Without this, a 350 ms
    # grace-then-fallback timer races the actual WAV arrival and you
    # get DOUBLE TTS (browser voice + macOS `say` voice both speaking
    # the same line). Including it on the `ready` frame lets the
    # client pick exactly one TTS source for the whole session.
    ready_payload = {
        "type": "ready",
        "tts_available": HAS_SAY,
        "intent_loaded": intent_holder.loaded,
        "whisper_loaded": whisper_holder.loaded,
    }
    try:
        # Tell the client whether the model is still warming up.
        if holder.loaded:
            await send(ready_payload)
        else:
            await send({"type": "loading"})
            await holder.ensure_loaded()
            if holder.load_error:
                await send({
                    "type": "error",
                    "code": ExitCode.SETUP_MODEL_LOAD_FAILED,
                    "message": holder.load_error,
                })
                return
            # Re-snapshot the secondary holders' state — they may have
            # finished loading while we waited on Gemma 4.
            ready_payload["intent_loaded"] = intent_holder.loaded
            ready_payload["whisper_loaded"] = whisper_holder.loaded
            await send(ready_payload)

        # Wait for the client to start the session.
        first_text = await ws.receive_text()
        first = json.loads(first_text)
        if first.get("type") != "start_session":
            await send({
                "type": "error",
                "code": ExitCode.CONFIG_INVALID,
                "message": f"expected start_session, got {first.get('type')!r}",
            })
            return

        await send_current_drill()

        while drill_idx < len(drills_list) and not rest_called:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError as exc:
                await send({
                    "type": "error",
                    "code": ExitCode.CONFIG_INVALID,
                    "message": f"bad JSON: {exc}",
                })
                continue
            mtype = msg.get("type")

            if mtype == "command":
                action = msg.get("action")
                if action == "rest":
                    rest_called = True
                    break
                if action == "skip":
                    drill_idx += 1
                    retries_for_drill = 0
                    if drill_idx < len(drills_list):
                        await send({"type": "advance"})
                        await send_current_drill()
                    continue
                if action == "repeat_prompt":
                    await send_current_drill()
                    continue
                continue  # pragma: no cover - unknown command, treat as no-op

            if mtype in ("intent", "intent_audio"):
                # Free-form utterance routed by FunctionGemma 270M.
                # Two input modes:
                #   - "intent":       client already has text (typed or
                #                     transcribed locally)
                #   - "intent_audio": client sent raw PCM; we transcribe
                #                     on-device with Cactus Whisper here
                #                     so audio never leaves the machine.
                # Both end up calling the same intent_mod.classify().
                transcript = ""
                transcribe_latency_ms: int | None = None
                if mtype == "intent_audio":
                    if not whisper_holder.loaded:
                        await send({
                            "type": "error",
                            "code": ExitCode.RUNTIME_MODEL_FAILED,
                            "message": (
                                "Whisper STT not loaded yet — "
                                "type the command or wait for warmup."
                            ),
                        })
                        continue
                    sample_rate = msg.get("sample_rate", coach.SR)
                    if sample_rate != coach.SR:
                        await send({
                            "type": "error",
                            "code": ExitCode.CONFIG_INVALID,
                            "message": f"sample_rate {sample_rate} != {coach.SR}",
                        })
                        continue
                    try:
                        cmd_pcm = base64.b64decode(msg.get("pcm_b64", ""))
                    except Exception as exc:
                        await send({
                            "type": "error",
                            "code": ExitCode.CONFIG_INVALID,
                            "message": f"bad pcm_b64: {exc}",
                        })
                        continue
                    if not cmd_pcm:
                        await send({
                            "type": "error",
                            "code": ExitCode.RUNTIME_AUDIO_FAILED,
                            "message": "empty command audio",
                        })
                        continue
                    try:
                        transcript, transcribe_latency_ms = await asyncio.to_thread(
                            whisper_holder.transcribe, cmd_pcm,
                        )
                    except Exception as exc:
                        log.error("whisper transcribe failed: %s", exc)
                        await send({
                            "type": "error",
                            "code": ExitCode.RUNTIME_MODEL_FAILED,
                            "message": f"whisper failed: {exc}",
                        })
                        continue
                    utterance = transcript
                else:
                    utterance = str(msg.get("utterance", ""))
                    transcript = utterance

                result = await asyncio.to_thread(
                    intent_mod.classify, utterance, intent_holder.classifier(),
                )
                payload = result.to_payload()
                payload["intent_model_loaded"] = intent_holder.loaded
                payload["transcript"] = transcript
                # transcribe_latency_ms is None for the typed path —
                # frontend hides the "STT" chip in that case.
                payload["transcribe_latency_ms"] = transcribe_latency_ms
                payload["transcribe_source"] = (
                    "whisper" if mtype == "intent_audio" else "client"
                )
                await send({"type": "intent_result", **payload})

                if result.intent is intent_mod.Intent.REST:
                    rest_called = True
                    break
                if result.intent is intent_mod.Intent.SKIP:
                    drill_idx += 1
                    retries_for_drill = 0
                    if drill_idx < len(drills_list):
                        await send({"type": "advance"})
                        await send_current_drill()
                    continue
                if result.intent is intent_mod.Intent.REPEAT:
                    await send_current_drill()
                    continue
                # Intent.NONE: do nothing. The client decides whether to
                # tell the user "didn't catch a command" or stay silent.
                continue

            if mtype != "audio":
                await send({
                    "type": "error",
                    "code": ExitCode.CONFIG_INVALID,
                    "message": f"unexpected message type: {mtype!r}",
                })
                continue

            sample_rate = msg.get("sample_rate", coach.SR)
            if sample_rate != coach.SR:
                await send({
                    "type": "error",
                    "code": ExitCode.CONFIG_INVALID,
                    "message": f"sample_rate {sample_rate} != {coach.SR}",
                })
                continue

            try:
                pcm = base64.b64decode(msg.get("pcm_b64", ""))
            except Exception as exc:
                await send({
                    "type": "error",
                    "code": ExitCode.CONFIG_INVALID,
                    "message": f"bad pcm_b64: {exc}",
                })
                continue

            min_bytes = 2 * coach.MIN_SPEECH_MS * coach.SR // 1000
            if len(pcm) < min_bytes:
                await send({
                    "type": "error",
                    "code": ExitCode.RUNTIME_AUDIO_FAILED,
                    "message": (
                        f"audio too short: {len(pcm)} bytes "
                        f"(<{min_bytes} = {coach.MIN_SPEECH_MS}ms)"
                    ),
                })
                continue

            dbfs, duration_s = _pcm_bytes_to_dbfs(pcm)
            dbfs_seen.append(dbfs)
            await send({
                "type": "metrics",
                "dbfs": round(dbfs, 1) if math.isfinite(dbfs) else None,
                "duration_s": round(duration_s, 2),
            })
            # First step the user sees: "we have your audio, Gemma is
            # chewing on it." Sent BEFORE the to_thread call so the
            # spinner has something to label even on the slowest
            # machines (where ~10s can pass before the first token
            # streams back).
            await send({
                "type": "thinking",
                "step": "analyzing_audio",
                "label": "Analyzing your audio with Gemma 4…",
            })

            d = drills_list[drill_idx]
            sys_prompt = coach.build_prompt(d, dbfs, duration_s)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ""},
            ]
            try:
                holder.cactus.cactus_reset(holder.model)
            except Exception as exc:
                log.debug("cactus_reset raised %s", exc)

            collector = coach._TokenCollector()
            # Wrap the collector so the first streamed token flips the
            # UI from "analyzing audio" → "generating response". The
            # callback runs on Cactus's worker thread (we're inside
            # asyncio.to_thread), so we hop back onto the loop with
            # run_coroutine_threadsafe; doing the WS send directly from
            # the worker thread would race the asyncio writer.
            loop = asyncio.get_running_loop()
            first_token_signalled = False

            def progress_collector(text: str, tid: int) -> None:
                nonlocal first_token_signalled
                collector(text, tid)
                if first_token_signalled or not text:
                    return
                first_token_signalled = True
                asyncio.run_coroutine_threadsafe(
                    send({
                        "type": "thinking",
                        "step": "generating_response",
                        "label": "Generating coaching response…",
                    }),
                    loop,
                )

            t0 = time.monotonic()
            try:
                envelope = await asyncio.to_thread(
                    chat._cactus_complete_audio,
                    holder.cactus, holder.model,
                    json.dumps(messages),
                    holder.options_json,
                    pcm,
                    progress_collector,
                )
            except Exception as exc:
                log.error("model call failed on %s/%d: %s",
                          d.stage, d.index, exc)
                coach._append_jsonl(session_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drill": asdict(d),
                    "achieved_dbfs": dbfs,
                    "duration_s": duration_s,
                    "error": str(exc),
                })
                await send({
                    "type": "error",
                    "code": ExitCode.RUNTIME_MODEL_FAILED,
                    "message": str(exc),
                })
                continue

            latency = time.monotonic() - t0
            streamed = collector.text()
            raw_for_parse = streamed or envelope
            log.debug(
                "drill=%s/%d latency=%.2fs streamed=%dB envelope=%dB",
                d.stage, d.index, latency, len(streamed), len(envelope),
            )

            # Two cleanup beats before we move on:
            # 1. Yield so any `generating_response` frame scheduled from
            #    the token-callback thread drains BEFORE we send the
            #    next step (otherwise the UI sees parsing_response, then
            #    generating_response arrives late and rewinds the
            #    progress list).
            # 2. If the model never streamed a token (envelope-only
            #    completion, no callback fired), emit the frame here so
            #    the UI still shows the step transition.
            await asyncio.sleep(0)
            if not first_token_signalled:
                await send({
                    "type": "thinking",
                    "step": "generating_response",
                    "label": "Generating coaching response…",
                })

            await send({
                "type": "thinking",
                "step": "parsing_response",
                "label": "Parsing model output…",
            })

            valid = coach.validate_coach_json(coach.parse_coach_json(raw_for_parse))
            if not valid:
                json_failures += 1
                coach._append_jsonl(session_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "drill": asdict(d),
                    "achieved_dbfs": dbfs,
                    "duration_s": duration_s,
                    "model_latency_s": round(latency, 3),
                    "streamed_reply": streamed,
                    "envelope_reply": envelope,
                    "parse_error": True,
                })
                await send({
                    "type": "error",
                    "code": ExitCode.RUNTIME_BAD_JSON,
                    "message": "model produced invalid JSON",
                })
                # Skip past this drill so the loop keeps moving.
                drill_idx += 1
                retries_for_drill = 0
                if drill_idx < len(drills_list):
                    await send({"type": "advance"})
                    await send_current_drill()
                continue

            heard = (valid.get("heard") or "").strip()
            ack = (valid.get("ack") or "").strip()
            feedback = (valid.get("feedback") or "").strip()
            action = valid["next_action"]
            mo = valid.get("metrics_observed", {}) or {}
            matched = bool(mo.get("matched_prompt", True))

            await send({
                "type": "coach",
                "heard": heard,
                "matched_prompt": matched,
                "ack": ack,
                "feedback": feedback,
                "next_action": action,
                "metrics_observed": mo,
                "latency_s": round(latency, 2),
            })

            # Server-side TTS: render the coach's spoken line locally
            # via macOS `say` and ship the WAV to the client. Keeps
            # the entire voice path (in AND out) on this machine — no
            # Chrome-cloud TTS, no platform-dependent voice quality.
            # If `say` isn't available (Linux/CI), we just skip this
            # frame and the frontend falls back to speechSynthesis.
            spoken = ". ".join(p for p in (ack, feedback) if p).strip()
            if spoken and HAS_SAY:
                # Last visible step before audio actually starts playing.
                # Gated on HAS_SAY so we don't promise voice synthesis on
                # a host (Linux/CI) where `say` is missing.
                await send({
                    "type": "thinking",
                    "step": "synthesizing_voice",
                    "label": "Synthesizing voice with macOS say…",
                })
                wav = await asyncio.to_thread(_say_to_wav, spoken)
                if wav is not None:
                    await send({
                        "type": "audio_reply",
                        "wav_b64": base64.b64encode(wav).decode("ascii"),
                        "source": "macos_say",
                    })

            coach._append_jsonl(session_path, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "drill": asdict(d),
                "achieved_dbfs": dbfs,
                "duration_s": duration_s,
                "model_latency_s": round(latency, 3),
                "coach": valid,
            })

            if action == "rest":
                rest_called = True
                break

            if (
                action == "retry"
                and retries_for_drill < coach.MAX_RETRIES_PER_DRILL
            ):
                retries_for_drill += 1
                retries_total += 1
                await send({"type": "retry"})
                await send_current_drill()
                continue

            advanced += 1
            drill_idx += 1
            retries_for_drill = 0
            if drill_idx < len(drills_list):
                await send({"type": "advance"})
                await send_current_drill()

        avg = (
            (sum(x for x in dbfs_seen if math.isfinite(x))
             / sum(1 for x in dbfs_seen if math.isfinite(x)))
            if any(math.isfinite(x) for x in dbfs_seen)
            else None
        )
        await send({
            "type": "session_done",
            "summary": {
                "advanced": advanced,
                "total": len(drills_list),
                "retries": retries_total,
                "avg_dbfs": round(avg, 1) if avg is not None else None,
                "json_failures": json_failures,
                "rest_called": rest_called,
                "session_log": str(session_path),
            },
        })

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as exc:  # pragma: no cover - last-resort safety net
        log.exception("ws_coach error")
        try:
            await send({
                "type": "error",
                "code": ExitCode.RUNTIME_MODEL_FAILED,
                "message": str(exc),
            })
        except Exception:
            pass
