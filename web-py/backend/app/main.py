"""FastAPI server for the Voice Coach web app.

Single WebSocket endpoint `/ws/coach` drives the drill loop. The browser
handles mic capture (16 kHz mono Int16 PCM, base64-encoded over the wire)
and TTS via `window.speechSynthesis`; the server runs Gemma 4 inference
via Cactus on the user's machine.

We reuse cli/ wholesale:
  - cli/_log.py            structured logging
  - cli/_exit.py           typed exit codes (used as error codes over WS)
  - cli/content.py         drill content set (Drill dataclass, default_drill_set)
  - cli/coach.py           prompt builder, JSON parser/validator, dBFS helper,
                           session-log helpers, _TokenCollector
  - cli/chat.py            ensure_lib_discoverable, ensure_model,
                           _cactus_complete_audio (the FFI helper)

Wire format
-----------
Client -> Server:
  { "type": "start_session" }
  { "type": "audio", "pcm_b64": "<base64 int16 PCM>", "sample_rate": 16000 }
  { "type": "command", "action": "skip" | "rest" | "repeat_prompt" }

Server -> Client:
  { "type": "loading" }
  { "type": "ready" }
  { "type": "drill", stage, index, prompt, note, target_dbfs, total, position }
  { "type": "metrics", dbfs, duration_s }
  { "type": "thinking" }
  { "type": "coach", heard, matched_prompt, ack, feedback,
                     next_action, metrics_observed, latency_s }
  { "type": "advance" | "retry" | "rest" }
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
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import _log                            # noqa: E402  (cli module on sys.path)
from _exit import ExitCode             # noqa: E402
import chat                            # noqa: E402
import coach                           # noqa: E402
import content                         # noqa: E402

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware           # noqa: E402

log = _log.get("server")

DEFAULT_MODEL_ID = "google/gemma-4-e2b-it"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 256


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

    async def ensure_loaded(self) -> None:
        if self.loaded or self.load_error:
            return
        async with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                # Heavy work runs in a thread so it doesn't block the loop.
                await asyncio.to_thread(self._load_blocking)
            except Exception as exc:  # pragma: no cover - reported to client
                log.exception("model load failed")
                self.load_error = str(exc)

    def _load_blocking(self) -> None:
        chat.ensure_lib_discoverable()
        self.weights_path = chat.ensure_model(DEFAULT_MODEL_ID)
        import cactus  # type: ignore  # late import: needs dylib + bindings
        self.cactus = cactus
        log.info("loading model from %s", self.weights_path)
        self.model = cactus.cactus_init(str(self.weights_path), None, False)
        log.info("model ready")


holder = ModelHolder()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Kick off model load eagerly so the first WS connection finds it ready.
    asyncio.create_task(holder.ensure_loaded())
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
    }


@app.get("/api/drills")
async def drills() -> list[dict]:
    return [asdict(d) for d in content.default_drill_set()]


def _pcm_bytes_to_dbfs(pcm: bytes) -> tuple[float, float]:
    """Return (dbfs, duration_s) for raw int16 little-endian PCM at 16 kHz."""
    if not pcm:
        return -math.inf, 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
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
        await send({
            "type": "drill",
            "stage": d.stage,
            "index": d.index,
            "prompt": d.prompt,
            "note": d.note,
            "target_dbfs": d.target_dbfs,
            "total": len(drills_list),
            "position": drill_idx,
        })

    try:
        # Tell the client whether the model is still warming up.
        if holder.loaded:
            await send({"type": "ready"})
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
            await send({"type": "ready"})

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
            await send({"type": "thinking"})

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
            t0 = time.monotonic()
            try:
                envelope = await asyncio.to_thread(
                    chat._cactus_complete_audio,
                    holder.cactus, holder.model,
                    json.dumps(messages),
                    holder.options_json,
                    pcm,
                    collector,
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
    except Exception as exc:
        log.exception("ws_coach error")
        try:
            await send({
                "type": "error",
                "code": ExitCode.RUNTIME_MODEL_FAILED,
                "message": str(exc),
            })
        except Exception:
            pass
