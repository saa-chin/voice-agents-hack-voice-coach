"""Shared TTS helpers reused by chat (voice mode) and the coach loop.

The single non-trivial bit is `split_sentences`: a streaming-friendly
splitter that pulls out complete sentences from a growing buffer of
streamed model tokens so we can fire TTS *as soon as the first sentence
boundary appears*, not after the whole reply has finished decoding.

This is the difference between perceived latency = "first sentence" and
perceived latency = "full JSON". With Gemma 4 doing ~60 decode tps and
~80-token replies, that's ~1 s vs ~2 s.

Pure functions only — no audio backend, no I/O. The CLI uses macOS
`say`; the web server renders each sentence to a WAV via `say` and
ships it as `audio_reply_chunk` frames.
"""
from __future__ import annotations


_SENTENCE_END_CHARS = ".!?"
_TRAILING_PUNCT_AFTER_END = ".!?\"')]"
_TRAILING_BREAK = " \n\t"


def split_sentences(text: str) -> tuple[list[str], str]:
    """Pull complete sentences off the front of `text`.

    Returns (done, leftover) where:
      - done      = list of complete sentences (each ends with .!? + closing
                    quote/paren cluster, leading/trailing whitespace stripped)
      - leftover  = the in-progress fragment after the last complete sentence

    A sentence is "complete" when a terminal `.!?` is followed by a space,
    newline, tab, or end-of-string. Trailing closing characters
    (additional `.!?`, `"`, `'`, `)`, `]`) are pulled into the same
    sentence so we don't break up `"Right!"` or `(yes.)`.

    Pure function — safe to call repeatedly as new tokens arrive.
    """
    out: list[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        if ch in _SENTENCE_END_CHARS:
            j = i + 1
            while j < n and text[j] in _TRAILING_PUNCT_AFTER_END:
                buf += text[j]
                j += 1
            if j >= n or text[j] in _TRAILING_BREAK:
                out.append(buf.strip())
                buf = ""
                if j < n:
                    j += 1
                i = j
                continue
            i = j
            continue
        i += 1
    return out, buf


__all__ = ["split_sentences"]
