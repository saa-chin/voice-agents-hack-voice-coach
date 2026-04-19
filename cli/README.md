# Cactus CLI

Local on-device chat REPL using [Cactus](https://docs.cactuscompute.com/).
Two modes:

- **text** — type, Gemma 4 streams a reply
- **voice** — continuous mic capture; raw PCM goes **directly into Gemma 4**
  (audio-native, no Whisper). Reply is spoken via macOS `say`.

Everything runs on-device.

## Quick start

From the repo root:

```bash
./run-cli                # interactive — pick text or voice
./run-cli --text         # text chat
./run-cli --voice        # continuous voice chat
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

- 16 kHz mono mic capture in 100 ms frames
- Energy-based VAD (RMS threshold). Speech start triggers capture; ~900 ms
  of silence ends the turn (or 28s hard cap)
- The whole utterance's PCM is sent to Gemma 4 in one call
- Tokens stream to stdout; sentences are spoken via `say` as they complete
- Ctrl+C exits

## Manual run

```bash
python3.14 cli/chat.py                              # text
python3.14 cli/chat.py --mode voice                 # voice
python3.14 cli/chat.py --model google/gemma-3-270m-it
python3.14 cli/chat.py --voice-name Samantha
```

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
