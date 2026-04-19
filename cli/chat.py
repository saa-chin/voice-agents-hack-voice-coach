#!/usr/bin/env python3.14
"""Interactive on-device chat REPL backed by Cactus."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BREW_PREFIX = Path("/opt/homebrew")
CACTUS_LIB = BREW_PREFIX / "opt/cactus/lib/libcactus.dylib"
CACTUS_WEIGHTS_DIR = BREW_PREFIX / "opt/cactus/libexec/weights"
EXPECTED_LIB = BREW_PREFIX / "lib/cactus/build/libcactus.dylib"


def ensure_lib_discoverable() -> None:
    # cactus.py hardcodes <site-packages>/../../cactus/build/libcactus.dylib.
    # Symlink the brew dylib so the loader finds it.
    if EXPECTED_LIB.exists() or EXPECTED_LIB.is_symlink():
        return
    if not CACTUS_LIB.exists():
        sys.exit(f"libcactus.dylib not found at {CACTUS_LIB}. Install: brew install cactus-compute/cactus/cactus")
    EXPECTED_LIB.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_LIB.symlink_to(CACTUS_LIB)


def ensure_model(model_id: str) -> Path:
    weights = CACTUS_WEIGHTS_DIR / model_id.split("/")[-1]
    if weights.exists() and any(weights.iterdir()):
        return weights
    print(f"Downloading {model_id} ...", file=sys.stderr)
    subprocess.run(["cactus", "download", model_id], check=True)
    return weights


def chat(model_id: str, system: str, temperature: float, max_tokens: int) -> None:
    ensure_lib_discoverable()
    weights = ensure_model(model_id)

    import cactus

    print(f"Loading {model_id} from {weights} ...", file=sys.stderr)
    model = cactus.cactus_init(str(weights), None, False)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    options = json.dumps({"temperature": temperature, "max_tokens": max_tokens})

    print("Type your message. Commands: /reset clears history, /quit exits.\n")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user in ("/quit", "/exit"):
                break
            if user == "/reset":
                messages = [m for m in messages if m["role"] == "system"]
                cactus.cactus_reset(model)
                print("(history cleared)\n")
                continue

            messages.append({"role": "user", "content": user})

            print("bot> ", end="", flush=True)
            chunks: list[str] = []

            def on_token(text: str, _token_id: int) -> None:
                chunks.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

            cactus.cactus_complete(
                model,
                json.dumps(messages),
                options,
                None,
                on_token,
            )
            reply = "".join(chunks)
            print()
            messages.append({"role": "assistant", "content": reply})
    finally:
        cactus.cactus_destroy(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Cactus chat CLI")
    parser.add_argument("--model", default="google/gemma-4-E4B-it", help="HF model id (default: gemma-4-E4B-it)")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    chat(args.model, args.system, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
