# Voice Coach — Web App

Browser UI for the same drill-driven speech coach that ships in `cli/`. The
backend is a thin FastAPI WebSocket server that reuses the CLI's prompt builder,
JSON parser, and Cactus FFI helper. The frontend is an Astro page with a single
React island that handles mic capture (`MediaRecorder` → 16 kHz mono Int16 PCM)
and TTS (`window.speechSynthesis`).

Everything runs on your machine. Audio never leaves it.

## Quick start

From the repo root:

```bash
./run-web                      # default ports: backend 8765, frontend 4321
./run-web --no-open            # don't auto-open the browser
./run-web --build              # serve a production build instead of dev
./run-web --backend-port 9000  # override backend port
./run-web --log-level DEBUG    # verbose backend logs
```

`run-web` does everything end-to-end:

1. Verifies `python3.14`, `cactus`, `node`, `npm`, the dylib, and the bindings.
2. Symlinks `libcactus.dylib` where the bindings expect it.
3. Downloads the chat model (`google/gemma-4-e2b-it`) on first run.
4. Installs Python deps (`fastapi`, `uvicorn`, `websockets`) if missing.
5. Installs Node deps (`npm install` in `frontend/`) if missing.
6. **Frees the backend + frontend ports** — terminates any process still
   listening on 8765 / 4321 (e.g. an orphan from a prior aborted run) so
   `run-web` is always idempotent. Process name + PID are logged before
   the kill so you can spot the rare case it's not what you expected.
7. Boots the backend (uvicorn) and frontend (Astro dev) in the background.
8. Waits for both to be reachable, then opens `http://127.0.0.1:4321`.
9. On Ctrl+C, kills both child processes cleanly.

Logs land in `.run-web-logs/{backend,frontend}.log` at the repo root.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Browser (Astro + React island)                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CoachApp.tsx                                                      │  │
│  │   - WebSocket → ws://127.0.0.1:8765/ws/coach                       │  │
│  │   - MediaRecorder + AudioContext → 16 kHz mono Int16 PCM           │  │
│  │   - speechSynthesis.speak(ack + feedback)                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  ▲                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │ WebSocket: JSON frames
┌──────────────────────────────────┼───────────────────────────────────────┐
│  Backend (FastAPI + uvicorn) on the same machine                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  app/main.py                                                        │  │
│  │   - /ws/coach: drives the drill loop                                │  │
│  │   - imports cli/coach.py, cli/content.py, cli/chat.py               │  │
│  │     (re-uses prompt builder, JSON parser, FFI helper, content set)  │  │
│  │   - calls Cactus on a thread (await asyncio.to_thread(...))         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│              libcactus.dylib + Gemma 4 E2B (INT4) on CPU/NPU             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Wire format (`/ws/coach`)

```text
Client → Server
  { type: "start_session" }
  { type: "audio", pcm_b64: "<base64 int16 LE>", sample_rate: 16000 }
  { type: "command", action: "skip" | "rest" | "repeat_prompt" }

Server → Client
  { type: "loading" }                                  # model still warming up
  { type: "ready" }                                    # send start_session now
  { type: "drill", stage, index, prompt, note,
                   target_dbfs, total, position }
  { type: "metrics", dbfs, duration_s }                # after audio received
  { type: "thinking" }                                 # inference in flight
  { type: "coach", heard, matched_prompt, ack,
                   feedback, next_action,
                   metrics_observed, latency_s }
  { type: "advance" | "retry" | "rest" }               # cue before next drill
  { type: "session_done", summary: { advanced, total,
                                      retries, avg_dbfs,
                                      json_failures,
                                      rest_called,
                                      session_log } }
  { type: "error", code, message }                     # code uses cli/_exit.py
```

## Customising

- **Drill content**: edit `cli/content.py`. Both the CLI and the web app read
  from the same `default_drill_set()`.
- **System prompt**: edit `cli/coach.py::COACH_SYSTEM_TEMPLATE`.
- **Backend WS URL**: set `PUBLIC_BACKEND_WS_URL` in
  `web-py/frontend/.env` to point the browser at a different host/port.
  Example for serving over LAN:
  ```
  PUBLIC_BACKEND_WS_URL=ws://192.168.1.10:8765/ws/coach
  ```
- **Session log location**: set `VOICE_COACH_SESSION_DIR=/path` before
  launching `run-web`. Defaults to `~/.voice-coach/sessions/`.

## Manual run (without run-web)

```bash
# terminal 1 — backend
cd web-py/backend
VOICE_COACH_LOG_LEVEL=DEBUG python3.14 -m uvicorn app.main:app --port 8765

# terminal 2 — frontend
cd web-py/frontend
npm install         # first time only
npm run dev
```

Visit `http://127.0.0.1:4321`.

## Browser support

- Chrome, Edge, Safari (macOS 14+), Firefox: all work.
- Requires HTTPS or `127.0.0.1`/`localhost` origin for `getUserMedia`. The
  Astro dev server binds to `127.0.0.1` so this is satisfied automatically.
- TTS quality varies by browser. Safari uses macOS system voices (best on
  Mac). Chrome falls back to its bundled "Google US English" voice.

## Exit codes

The `run-web` script and the WebSocket `error` frames use the same numeric
scheme as the CLI (`cli/_exit.py`):

| Code | Meaning |
| --- | --- |
| 0   | success |
| 10  | required tool missing (`python3.14`, `cactus`, `node`, `npm`) |
| 11  | `libcactus.dylib` missing |
| 12  | Cactus Python bindings missing |
| 13  | Python dep install failed (fastapi/uvicorn) or `npm install` failed |
| 20  | model download or production build failed |
| 21  | backend or frontend exited before becoming reachable |
| 22  | dylib symlink failed |
| 30  | audio-related runtime error (e.g. clip too short) |
| 31  | model inference failed |
| 32  | model returned malformed JSON |
| 40  | invalid args / wire-format violation |
| 130 | Ctrl+C |

## What this reuses from the CLI

- `cli/coach.py`: `build_prompt`, `parse_coach_json`, `validate_coach_json`,
  `rms_to_dbfs`, `_TokenCollector`, `_open_session_log`, `_append_jsonl`,
  `MAX_RETRIES_PER_DRILL`, `SR`, `MIN_SPEECH_MS`.
- `cli/chat.py`: `ensure_lib_discoverable`, `ensure_model`,
  `_cactus_complete_audio`.
- `cli/content.py`: `Drill`, `default_drill_set`.
- `cli/_log.py`, `cli/_exit.py`: structured logging + typed exit codes.

The only thing the server reimplements is mic capture (browser does it) and
TTS (browser does it). The drill state machine, prompt contract, JSON parser,
and session logging are 100% shared with the CLI.
