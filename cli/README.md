# Cactus CLI

Local on-device chat REPL using [Cactus](https://docs.cactuscompute.com/).
Runs a small LLM on your Mac via Cactus FFI bindings.

## Setup

```bash
brew install cactus-compute/cactus/cactus
cactus download google/gemma-4-E4B-it
```

The Homebrew install ships Python 3.14 bindings in
`/opt/homebrew/lib/python3.14/site-packages/cactus.py`. They expect the dylib
at `/opt/homebrew/lib/cactus/build/libcactus.dylib`; the script symlinks the
brew dylib there on first run.

## Run

```bash
python3.14 chat.py
python3.14 chat.py --model google/gemma-3-270m-it --system "You are terse."
```

Commands inside the REPL:

- `/reset` — clear chat history and KV cache
- `/quit` — exit

## Models

Default: `google/gemma-4-E4B-it` (~4 GB, multimodal-capable). Override with `--model`.

Smaller picks for faster load / lower RAM:

- `google/gemma-3-270m-it` (~150 MB)
- `LiquidAI/LFM2-350M`
- `Qwen/Qwen3-0.6B`

See `cactus --help` for the full list.
