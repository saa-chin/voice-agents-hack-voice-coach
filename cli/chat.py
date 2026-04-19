#!/usr/bin/env python3.14
"""On-device chat REPL backed by Cactus.

Two modes:
  text  — typed REPL into Gemma 4 (text only).
  voice — continuous mic capture; the captured PCM is sent **directly** to
          Gemma 4 (audio-native, no Whisper). Gemma 4 reasons over the raw
          audio (tone, pace, hesitation) and replies with text that is then
          spoken via macOS `say`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


BREW_PREFIX = Path("/opt/homebrew")
CACTUS_LIB = BREW_PREFIX / "opt/cactus/lib/libcactus.dylib"
CACTUS_WEIGHTS_DIR = BREW_PREFIX / "opt/cactus/libexec/weights"
EXPECTED_LIB = BREW_PREFIX / "lib/cactus/build/libcactus.dylib"


def ensure_lib_discoverable() -> None:
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
    if not (weights.exists() and any(weights.iterdir())):
        sys.exit(f"Download finished but weights still missing at {weights}")
    return weights


# ----------------------------------------------------------------------------
# Text chat
# ----------------------------------------------------------------------------

def text_chat(model_id: str, system: str, temperature: float, max_tokens: int) -> None:
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


# ----------------------------------------------------------------------------
# Voice chat — Gemma 4 audio-native (no Whisper)
# ----------------------------------------------------------------------------

VOICE_SR = 16000
VOICE_FRAME_MS = 100
VOICE_FRAME_SAMPLES = VOICE_SR * VOICE_FRAME_MS // 1000  # 1600
VOICE_SILENCE_RMS = 350.0          # int16 RMS threshold for "speech"
VOICE_END_OF_TURN_MS = 900         # silence after speech to end a turn
VOICE_MIN_SPEECH_MS = 350          # ignore blips shorter than this
VOICE_MAX_UTTERANCE_MS = 28_000    # hard cap (Gemma 4 audio works to ~30s)

_SAY_LOCK = threading.Lock()
_SAY_PROCESS: subprocess.Popen | None = None


def speak_blocking(text: str, voice: str | None) -> None:
    global _SAY_PROCESS
    text = (text or "").strip()
    if not text or not shutil.which("say"):
        return
    args = ["say"]
    if voice:
        args += ["-v", voice]
    args += ["--", text]
    with _SAY_LOCK:
        _SAY_PROCESS = subprocess.Popen(args)
    try:
        _SAY_PROCESS.wait()
    except KeyboardInterrupt:
        if _SAY_PROCESS and _SAY_PROCESS.poll() is None:
            _SAY_PROCESS.terminate()
        raise
    finally:
        with _SAY_LOCK:
            _SAY_PROCESS = None


def stop_speaking() -> None:
    with _SAY_LOCK:
        if _SAY_PROCESS and _SAY_PROCESS.poll() is None:
            _SAY_PROCESS.terminate()


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Return (complete sentences, leftover). Splits on .!? + space/EOL."""
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(text):
        ch = text[i]
        buf += ch
        if ch in ".!?":
            j = i + 1
            while j < len(text) and text[j] in ".!?\"')]":
                buf += text[j]
                j += 1
            if j >= len(text) or text[j] in " \n\t":
                out.append(buf.strip())
                buf = ""
                if j < len(text):
                    j += 1
                i = j
                continue
            i = j
            continue
        i += 1
    return out, buf


def _cactus_complete_audio(
    cactus_mod, model_handle, messages_json: str, options_json: str,
    pcm_bytes: bytes, on_token,
) -> str:
    """Direct FFI call so PCM transfer is a single memcpy (fast).

    The shipped cactus.cactus_complete wrapper builds a c_uint8 array by
    iterating each byte in Python, which is O(N) Python work and stalls on
    ~1 MB utterances. Bypass it.
    """
    lib = cactus_mod._lib
    TokenCallback = cactus_mod.TokenCallback

    buf = ctypes.create_string_buffer(65536)

    if on_token is not None:
        def _bridge(token_bytes, token_id, _ud):
            on_token(
                token_bytes.decode("utf-8", errors="ignore") if token_bytes else "",
                token_id,
            )
        cb = TokenCallback(_bridge)
    else:
        cb = TokenCallback()

    arr_type = ctypes.c_uint8 * len(pcm_bytes)
    pcm_arr = arr_type.from_buffer_copy(pcm_bytes)
    pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))

    rc = lib.cactus_complete(
        model_handle,
        messages_json.encode("utf-8"),
        buf, len(buf),
        options_json.encode("utf-8"),
        None,
        cb, None,
        pcm_ptr, len(pcm_bytes),
    )
    if rc < 0:
        err = cactus_mod.cactus_get_last_error() or "Audio completion failed"
        raise RuntimeError(err)
    return buf.value.decode("utf-8", errors="ignore")


def voice_chat(
    model_id: str,
    system: str,
    temperature: float,
    max_tokens: int,
    voice: str | None,
) -> None:
    ensure_lib_discoverable()

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        sys.exit(
            "Voice mode requires `sounddevice` and `numpy`.\n"
            "Install: python3.14 -m pip install sounddevice numpy\n"
            "(Or use ./run-cli which installs them automatically.)"
        )

    weights = ensure_model(model_id)

    import cactus

    print(f"Loading chat model {model_id} ...", file=sys.stderr)
    model = cactus.cactus_init(str(weights), None, False)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    chat_options = json.dumps({"temperature": temperature, "max_tokens": max_tokens})

    end_silence_frames = max(1, VOICE_END_OF_TURN_MS // VOICE_FRAME_MS)
    min_speech_frames = max(1, VOICE_MIN_SPEECH_MS // VOICE_FRAME_MS)
    max_frames = VOICE_MAX_UTTERANCE_MS // VOICE_FRAME_MS

    print()
    print(f"Voice chat ready. Gemma 4 listens to your voice directly.")
    print(f"Pause ~{VOICE_END_OF_TURN_MS}ms to send. Ctrl+C exits.")
    print()

    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        stop_speaking()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not interrupted:
            print("listening… ", end="", flush=True)

            speech_started = False
            speech_frames = 0
            silence_frames = 0
            total_frames = 0
            captured: list[bytes] = []

            with sd.InputStream(
                samplerate=VOICE_SR,
                channels=1,
                dtype="int16",
                blocksize=VOICE_FRAME_SAMPLES,
            ) as mic:
                while True:
                    data, _overflow = mic.read(VOICE_FRAME_SAMPLES)
                    frame = np.asarray(data, dtype=np.int16).reshape(-1)
                    if frame.size == 0:
                        continue
                    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                    is_speech = rms > VOICE_SILENCE_RMS

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                            print("(hearing you) ", end="", flush=True)
                        speech_frames += 1
                        silence_frames = 0
                    elif speech_started:
                        silence_frames += 1

                    if speech_started:
                        captured.append(frame.tobytes())
                        total_frames += 1

                    if (
                        speech_started
                        and speech_frames >= min_speech_frames
                        and silence_frames >= end_silence_frames
                    ):
                        break
                    if total_frames >= max_frames:
                        print("(max length) ", end="", flush=True)
                        break

            print()
            if not captured or speech_frames < min_speech_frames:
                print("(no speech detected)\n")
                continue

            pcm_bytes = b"".join(captured)
            duration_s = len(pcm_bytes) / 2 / VOICE_SR
            print(f"you> [spoke {duration_s:.1f}s of audio]")

            # Gemma 4 takes audio attached to the latest user message.
            # Keep history light: store a placeholder for past audio turns.
            current_user = {"role": "user", "content": ""}
            messages.append(current_user)

            print("bot> ", end="", flush=True)
            buffer = ""
            chunks: list[str] = []
            sentences_to_speak: list[str] = []
            speaker_done = threading.Event()

            def speaker_loop():
                while True:
                    if not sentences_to_speak:
                        if speaker_done.is_set() and not sentences_to_speak:
                            return
                        time.sleep(0.05)
                        continue
                    sent = sentences_to_speak.pop(0)
                    speak_blocking(sent, voice)

            speaker_thread = threading.Thread(target=speaker_loop, daemon=True)
            speaker_thread.start()

            def on_token(token_text: str, _tid: int) -> None:
                nonlocal buffer
                chunks.append(token_text)
                sys.stdout.write(token_text)
                sys.stdout.flush()
                buffer += token_text
                done, leftover = _split_sentences(buffer)
                if done:
                    sentences_to_speak.extend(done)
                    buffer = leftover

            try:
                _cactus_complete_audio(
                    cactus, model,
                    json.dumps(messages),
                    chat_options,
                    pcm_bytes,
                    on_token,
                )
            except Exception as e:
                print(f"\n(model error: {e})")
                messages.pop()
                speaker_done.set()
                speaker_thread.join()
                continue

            reply = "".join(chunks).strip()
            tail = buffer.strip()
            if tail:
                sentences_to_speak.append(tail)
            speaker_done.set()
            speaker_thread.join()
            print()

            # Replace placeholder with stable text for history continuity.
            current_user["content"] = "[spoken audio]"
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n(stopping)")
    finally:
        stop_speaking()
        try:
            cactus.cactus_destroy(model)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Cactus chat CLI (text or voice)")
    parser.add_argument("--mode", choices=("text", "voice"), default="text")
    parser.add_argument("--model", default="google/gemma-4-e2b-it",
                        help="HF id of chat model (default: gemma-4-e2b-it; "
                             "voice mode requires a multimodal-audio model)")
    parser.add_argument("--voice-name", default=None,
                        help="macOS `say` voice name, e.g. Samantha, Daniel")
    parser.add_argument("--system",
                        default=("You are a concise, helpful voice assistant. "
                                 "The user is talking to you out loud. "
                                 "Reply naturally in 1-3 short sentences."))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "voice":
        voice_chat(
            args.model, args.system,
            args.temperature, args.max_tokens, args.voice_name,
        )
    else:
        text_chat(args.model, args.system, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
