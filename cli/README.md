# Cactus CLI

Local on-device chat REPL using [Cactus](https://docs.cactuscompute.com/).
Three modes:

- **text** — type, Gemma 4 streams a reply
- **voice** — continuous mic capture; raw PCM goes **directly into Gemma 4**
  (audio-native, no Whisper). Reply is spoken via macOS `say`.
- **coach** — drill-driven speech-coach session. Same audio path as voice,
  but Gemma 4 is prompted to reply in a structured JSON contract
  (`ack`, `feedback`, `next_action`, `metrics_observed`). Each turn is
  appended to a session log (JSONL) and a summary is printed at the end.

Everything runs on-device.

## Quick start

From the repo root:

```bash
./run-cli                # interactive — pick text, voice, or coach
./run-cli --text         # text chat
./run-cli --voice        # continuous voice chat
./run-cli --coach        # drill-driven speech-coach session
./run-cli --list-models  # list locally downloaded weights
```

`run-cli` does everything end-to-end:

- Verifies `cactus`, `python3.14`, dylib + Python bindings
- Symlinks `libcactus.dylib` where the bindings expect it
- Downloads the chat model (default `google/gemma-4-e2b-it`) if missing
- In voice mode: installs `sounddevice` + `numpy` + `cffi` (rebuilding from
  source if the brewed `cffi` is missing the C extension)

## Why Gemma 4 for voice (not Whisper + LLM)

Gemma 4 is multimodal-from-scratch with a 300M-parameter audio conformer
that feeds straight into the transformer's residual stream. It reasons over
the raw audio — tone, hesitation, emphasis — not just the transcript. One
model, one forward pass, ~0.3s end-to-end on a 30s clip on Apple Silicon.

So voice mode here is just: mic → PCM bytes → `cactus_complete(..., pcm_buffer=…)`
→ tokens → `say`.

## Voice mode behaviour

- 16 kHz mono mic capture in 50 ms frames
- Energy-based VAD (RMS threshold). Speech start triggers capture; ~600 ms
  of silence ends the turn (or 28s hard cap)
- The whole utterance's PCM is sent to Gemma 4 in one call
- Tokens stream to stdout; sentences are spoken via `say` as they complete
- Ctrl+C exits

## Coach mode behaviour

- Iterates a small built-in drill set: 3 warm-up vowels → 5 functional
  phrases → 2 conversation prompts (see `cli/content.py`)
- For each drill: the coach speaks the prompt, the same VAD-driven capture
  runs, average voiced loudness is computed in dBFS (honest, not faked SPL),
  and the audio + a structured system prompt is sent to Gemma 4
- Gemma 4 must reply with strict JSON:

  ```json
  {
    "ack": "Nice and clear.",
    "feedback": "Try a slightly louder breath on the first word.",
    "next_action": "advance",
    "metrics_observed": {
      "loudness_ok": true,
      "pitch_range_ok": false,
      "pace_ok": true,
      "articulation_ok": true
    }
  }
  ```

- `next_action` drives the loop: `retry` repeats (up to 2× per drill),
  `advance` moves on, `rest` ends the session early
- Each turn is appended to a JSONL session log at
  `~/.voice-coach/sessions/session-<UTC>.jsonl`
  (override with `VOICE_COACH_SESSION_DIR=...`)
- A short summary prints at the end: drills completed, retries, average
  loudness, JSON failures, log path

## Manual run

```bash
python3.14 cli/chat.py                              # text
python3.14 cli/chat.py --mode voice                 # voice
python3.14 cli/chat.py --mode coach                 # coach session
python3.14 cli/chat.py --mode coach --log-level DEBUG
python3.14 cli/chat.py --model google/gemma-3-270m-it
python3.14 cli/chat.py --voice-name Samantha
```

## Logs and exit codes

Logs go to **stderr**, prefixed with `voicecoach.<module>`. Tune verbosity
with `--log-level DEBUG|INFO|WARNING|ERROR` or the env var
`VOICE_COACH_LOG_LEVEL`.

Exit codes (defined in `cli/_exit.py`) let wrappers branch on *why* something
failed:

| Code | Meaning |
| --- | --- |
| 0   | success |
| 10  | required CLI tool missing (`cactus`) |
| 11  | `libcactus.dylib` missing |
| 12  | Cactus Python bindings missing |
| 13  | Python dep missing (sounddevice, numpy, coach module) |
| 14  | no usable audio input device |
| 20  | model download failed |
| 21  | `cactus_init` model load failed |
| 22  | dylib symlink failed |
| 30  | mic read failed mid-utterance |
| 31  | model inference failed |
| 32  | model returned malformed JSON (coach mode) |
| 40  | invalid CLI arguments |
| 130 | Ctrl+C |

## REPL commands (text mode)

- `/reset` — clear chat history and KV cache
- `/quit` — exit

## Models

Default: `google/gemma-4-e2b-it` (multimodal: text + vision + audio, ~6 GB
INT4). Voice mode requires a multimodal-audio model — use Gemma 4 E2B or E4B.

Smaller text-only options (won't work for voice):

- `google/gemma-3-270m-it`
- `LiquidAI/LFM2-350M`
- `Qwen/Qwen3-0.6B`

See `cactus list` for everything supported.

## Setup notes

```bash
brew install cactus-compute/cactus/cactus
brew install python@3.14
```

The Homebrew install ships Python 3.14 bindings in
`/opt/homebrew/lib/python3.14/site-packages/cactus.py`. They expect the dylib
at `/opt/homebrew/lib/cactus/build/libcactus.dylib`; `run-cli` (and `chat.py`
on first run) symlinks the brew dylib there.

If voice mode complains about `_cffi_backend`, the brewed `cffi` shim is
incomplete for Python 3.14. `run-cli` fixes this automatically; manual fix:

```bash
python3.14 -m pip install --break-system-packages --no-binary cffi cffi
```
