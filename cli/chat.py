#!/usr/bin/env python3.14
"""On-device chat REPL backed by Cactus.

Three modes:
  text  — typed REPL into Gemma 4 (text only).
  voice — continuous mic capture; the captured PCM is sent **directly** to
          Gemma 4 (audio-native, no Whisper). Gemma 4 reasons over the raw
          audio (tone, pace, hesitation) and replies with text that is then
          spoken via macOS `say`.
  coach — drill-driven speech practice. Same audio path as voice, but the
          model is instructed to reply in a structured JSON contract
          ({ack, feedback, next_action, metrics_observed}). See coach.py.

Exit codes are defined in `_exit.ExitCode`. Logs go to stderr via the
`voicecoach.*` logger family; tweak with --log-level or VOICE_COACH_LOG_LEVEL.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import _log
from _exit import ExitCode, die

log = _log.get("chat")


BREW_PREFIX = Path("/opt/homebrew")
CACTUS_LIB = BREW_PREFIX / "opt/cactus/lib/libcactus.dylib"
CACTUS_WEIGHTS_DIR = BREW_PREFIX / "opt/cactus/libexec/weights"
EXPECTED_LIB = BREW_PREFIX / "lib/cactus/build/libcactus.dylib"


def ensure_lib_discoverable() -> None:
    if EXPECTED_LIB.exists() or EXPECTED_LIB.is_symlink():
        return
    if not CACTUS_LIB.exists():
        die(
            ExitCode.ENV_MISSING_LIB,
            f"libcactus.dylib not found at {CACTUS_LIB}",
            "Install: brew install cactus-compute/cactus/cactus",
        )
    try:
        EXPECTED_LIB.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED_LIB.symlink_to(CACTUS_LIB)
    except OSError as exc:
        die(
            ExitCode.SETUP_LIB_LINK_FAILED,
            f"Failed to symlink {EXPECTED_LIB} -> {CACTUS_LIB}: {exc}",
            "Try manually: "
            f"sudo ln -sf {CACTUS_LIB} {EXPECTED_LIB}",
        )
    log.info("symlinked %s -> %s", EXPECTED_LIB, CACTUS_LIB)


def ensure_model(model_id: str) -> Path:
    weights = CACTUS_WEIGHTS_DIR / model_id.split("/")[-1]
    if weights.exists() and any(weights.iterdir()):
        return weights
    log.info("downloading model %s", model_id)
    try:
        subprocess.run(["cactus", "download", model_id], check=True)
    except FileNotFoundError:
        die(
            ExitCode.ENV_MISSING_TOOL,
            "`cactus` CLI not found on PATH",
            "Install: brew install cactus-compute/cactus/cactus",
        )
    except subprocess.CalledProcessError as exc:
        die(
            ExitCode.SETUP_DOWNLOAD_FAILED,
            f"`cactus download {model_id}` exited with {exc.returncode}",
            "Check internet, disk space, or HF auth (gated models).",
        )
    if not (weights.exists() and any(weights.iterdir())):
        die(
            ExitCode.SETUP_DOWNLOAD_FAILED,
            f"Download reported success but weights are missing at {weights}",
            f"Try: cactus download {model_id} --reconvert",
        )
    return weights


# ----------------------------------------------------------------------------
# Text chat
# ----------------------------------------------------------------------------

def text_chat(model_id: str, system: str, temperature: float, max_tokens: int) -> None:  # pragma: no cover - interactive REPL, exercised by hand via ./run-cli --text
    ensure_lib_discoverable()
    weights = ensure_model(model_id)

    try:
        import cactus
    except ImportError as exc:
        die(
            ExitCode.ENV_MISSING_BINDINGS,
            f"Cannot import `cactus` Python bindings: {exc}",
            "Reinstall: brew reinstall cactus-compute/cactus/cactus",
        )

    log.info("loading %s from %s", model_id, weights)
    try:
        model = cactus.cactus_init(str(weights), None, False)
    except Exception as exc:
        die(ExitCode.SETUP_MODEL_LOAD_FAILED, f"cactus_init failed: {exc}")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    options = json.dumps({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "confidence_threshold": 0.0,
    })

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
#
# Audio-format constants are owned by cli/coach.py and re-exported here under
# their chat-mode-prefixed names so the existing voice_chat() body keeps
# reading naturally. If you change a value, change it in coach.py.

import coach as _coach  # local alias to avoid clobbering the chat-history `coach`

VOICE_SR = _coach.SR
VOICE_FRAME_MS = _coach.FRAME_MS
VOICE_FRAME_SAMPLES = _coach.FRAME_SAMPLES
VOICE_SILENCE_RMS = _coach.SILENCE_RMS          # int16 RMS threshold for "speech"
VOICE_MIN_SPEECH_MS = _coach.MIN_SPEECH_MS      # ignore blips shorter than this
VOICE_MAX_UTTERANCE_MS = _coach.MAX_UTTERANCE_MS  # hard cap (Gemma 4 audio works to ~30s)
VOICE_PREROLL_MS = _coach.PREROLL_MS            # keep this much pre-speech audio for context
VOICE_POST_TTS_PAUSE_MS = _coach.POST_TTS_PAUSE_MS  # wait after TTS so we don't hear ourselves

# Chat-specific (intentionally diverges from coach mode):
# Conversational chat ends turns faster than coach drills (600 vs 800 ms),
# because we want snappy back-and-forth rather than tolerance for mid-thought
# pauses during phrase practice.
VOICE_END_OF_TURN_MS = 600
VOICE_HISTORY_TURNS = 4   # text-only history turns to keep for context

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
    except KeyboardInterrupt:  # pragma: no cover - timing-sensitive interactive path
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


def _cactus_complete_audio(  # pragma: no cover - direct ctypes FFI; needs libcactus.dylib loaded
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


def voice_chat(  # pragma: no cover - needs mic + Cactus + macOS `say`
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
    except ImportError as exc:
        die(
            ExitCode.ENV_MISSING_PYTHON_DEP,
            f"Voice mode requires sounddevice + numpy: {exc}",
            "Install: python3.14 -m pip install --break-system-packages sounddevice numpy",
            "Or use ./run-cli which installs them automatically.",
        )

    weights = ensure_model(model_id)

    try:
        import cactus
    except ImportError as exc:
        die(
            ExitCode.ENV_MISSING_BINDINGS,
            f"Cannot import `cactus` Python bindings: {exc}",
            "Reinstall: brew reinstall cactus-compute/cactus/cactus",
        )

    log.info("loading chat model %s", model_id)
    try:
        model = cactus.cactus_init(str(weights), None, False)
    except Exception as exc:
        die(ExitCode.SETUP_MODEL_LOAD_FAILED, f"cactus_init failed: {exc}")

    # text-only conversation history (paired user/assistant text turns)
    history: list[dict] = []
    sys_msg = {"role": "system", "content": system} if system else None

    # confidence_threshold=0 disables Cactus's automatic cloud handoff —
    # we want fully local, no warnings, no missed-key delays.
    chat_options = json.dumps({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "confidence_threshold": 0.0,
    })

    end_silence_frames = max(1, VOICE_END_OF_TURN_MS // VOICE_FRAME_MS)
    min_speech_frames = max(1, VOICE_MIN_SPEECH_MS // VOICE_FRAME_MS)
    max_frames = VOICE_MAX_UTTERANCE_MS // VOICE_FRAME_MS
    preroll_frames = max(0, VOICE_PREROLL_MS // VOICE_FRAME_MS)
    post_tts_pause_s = VOICE_POST_TTS_PAUSE_MS / 1000.0

    print()
    print("Voice chat ready. Gemma 4 listens to your voice directly.")
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
            preroll: list[bytes] = []

            with sd.InputStream(
                samplerate=VOICE_SR,
                channels=1,
                dtype="int16",
                blocksize=VOICE_FRAME_SAMPLES,
            ) as mic:
                # Drain any frames buffered while TTS was playing — those
                # are echoes of the bot's own voice through the speakers.
                drain_until = time.monotonic() + post_tts_pause_s
                while time.monotonic() < drain_until:
                    try:
                        mic.read(VOICE_FRAME_SAMPLES)
                    except Exception:
                        break

                while True:
                    data, _overflow = mic.read(VOICE_FRAME_SAMPLES)
                    frame = np.asarray(data, dtype=np.int16).reshape(-1)
                    if frame.size == 0:
                        continue
                    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                    is_speech = rms > VOICE_SILENCE_RMS
                    raw = frame.tobytes()

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                            captured.extend(preroll)
                            preroll.clear()
                            print("(hearing you) ", end="", flush=True)
                        speech_frames += 1
                        silence_frames = 0
                    elif speech_started:
                        silence_frames += 1
                    else:
                        preroll.append(raw)
                        if len(preroll) > preroll_frames:
                            preroll.pop(0)

                    if speech_started:
                        captured.append(raw)
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

            # Reset KV cache every turn. Otherwise the encoded audio from
            # previous turns lingers and the model may answer the wrong one.
            try:
                cactus.cactus_reset(model)
            except Exception:
                pass

            # Build a fresh message list: system prompt + last few text-only
            # turns of context + a single new user turn that the audio
            # attaches to. We don't keep prior user turns in history because
            # we have no transcript for them.
            turn_messages: list[dict] = []
            if sys_msg is not None:
                turn_messages.append(sys_msg)
            turn_messages.extend(history[-VOICE_HISTORY_TURNS * 2:])
            turn_messages.append({"role": "user", "content": ""})

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
                    json.dumps(turn_messages),
                    chat_options,
                    pcm_bytes,
                    on_token,
                )
            except Exception as e:
                print(f"\n(model error: {e})")
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

            # Keep only the assistant's reply for context. We pair it with a
            # short placeholder user turn so the model sees a normal chat
            # alternation when we replay history next turn.
            history.append({"role": "user", "content": "(spoke to you)"})
            history.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n(stopping)")
    finally:
        stop_speaking()
        try:
            cactus.cactus_destroy(model)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Coach mode — drill-driven JSON-structured speech practice
# ----------------------------------------------------------------------------

def coach_chat(  # pragma: no cover - shim; delegates to coach.coach_mode (full integration)
    model_id: str,
    voice: str | None,
    temperature: float,
    max_tokens: int,
) -> None:
    """Thin shim: load model + dylib, then hand off to coach.coach_mode()."""
    ensure_lib_discoverable()
    weights = ensure_model(model_id)

    try:
        import cactus
    except ImportError as exc:
        die(
            ExitCode.ENV_MISSING_BINDINGS,
            f"Cannot import `cactus` Python bindings: {exc}",
            "Reinstall: brew reinstall cactus-compute/cactus/cactus",
        )

    try:
        import coach as coach_mod
    except ImportError as exc:
        die(
            ExitCode.ENV_MISSING_PYTHON_DEP,
            f"Cannot import coach module: {exc}",
            "Make sure cli/coach.py and cli/content.py are present.",
        )

    coach_mod.coach_mode(
        cactus_module=cactus,
        cactus_complete_audio=_cactus_complete_audio,
        speak_blocking=speak_blocking,
        stop_speaking=stop_speaking,
        weights_path=weights,
        voice_name=voice,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Cactus chat CLI (text, voice, or coach)"
    )
    parser.add_argument(
        "--mode", choices=("text", "voice", "coach"), default="text",
        help="text REPL, open voice chat, or drill-driven coach session",
    )
    parser.add_argument("--model", default="google/gemma-4-e2b-it",
                        help="HF id of chat model (default: gemma-4-e2b-it; "
                             "voice/coach modes require a multimodal-audio model)")
    parser.add_argument("--voice-name", default=None,
                        help="macOS `say` voice name, e.g. Samantha, Daniel")
    parser.add_argument("--system",
                        default=(
                            "You are a friendly, natural-sounding voice "
                            "assistant. The user is speaking to you out loud, "
                            "so reply the way a person would in conversation: "
                            "warm, brief, and direct. Keep responses to 1-2 "
                            "short sentences unless asked for more. Never "
                            "echo the user's words back. Never describe what "
                            "you heard. Just answer."
                        ),
                        help="System prompt (text/voice modes only; coach mode "
                             "uses its own structured prompt)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (coach mode defaults to 0.4)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max generated tokens (coach mode bumps to 256)")
    parser.add_argument("--log-level", default=None,
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="Logger verbosity (overrides VOICE_COACH_LOG_LEVEL)")
    args = parser.parse_args()

    _log.configure(args.log_level)

    if args.mode == "voice":
        voice_chat(
            args.model, args.system,
            args.temperature, args.max_tokens, args.voice_name,
        )
    elif args.mode == "coach":
        # Coach mode tunes its own defaults if the user accepted the chat ones.
        temp = args.temperature if args.temperature != 0.7 else 0.4
        max_tok = args.max_tokens if args.max_tokens != 128 else 256
        coach_chat(args.model, args.voice_name, temp, max_tok)
    else:
        text_chat(args.model, args.system, args.temperature, args.max_tokens)


if __name__ == "__main__":  # pragma: no cover - script entry guard
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(ExitCode.USER_ABORT)
