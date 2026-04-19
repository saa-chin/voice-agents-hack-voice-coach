"""Drill content for the voice-coach CLI.

A small ordered set: warm-up vowels -> functional phrases -> conversation
prompts. The coach loop iterates this list once and lets the model's
`next_action` (retry / advance / rest) decide pacing.

`target_dbfs` is a relative loudness target, NOT a calibrated dB SPL value.
Calibrating mic input to room SPL needs a reference meter (R03 in the
implementation doc); until that lands we report dBFS and label it as such.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Drill:
    stage: str          # "warmup" | "phrase" | "conversation"
    index: int          # position within the stage (for display + logs)
    prompt: str         # what the patient should say
    note: str = ""      # optional cue shown alongside the prompt
    target_dbfs: float = -18.0


WARMUPS: list[Drill] = [
    Drill("warmup", 0, "Aaaah", "Hold for ~3 seconds, steady tone."),
    Drill("warmup", 1, "Eeeee", "Hold for ~3 seconds, even pitch."),
    Drill("warmup", 2, "Oooh",  "Hold for ~3 seconds, round mouth."),
]

PHRASES: list[Drill] = [
    Drill("phrase", 0, "Good morning, how are you?"),
    Drill("phrase", 1, "I would like a coffee, please."),
    Drill("phrase", 2, "Thank you, that was very helpful."),
    Drill("phrase", 3, "Could you say that again, please?"),
    Drill("phrase", 4, "My name is Alex and I live nearby."),
]

CONVERSATION: list[Drill] = [
    Drill("conversation", 0, "Tell me one good thing from your day."),
    Drill("conversation", 1, "What did you have for breakfast?"),
]


def default_drill_set() -> list[Drill]:
    """Default ordered set: warm-ups -> phrases -> conversation."""
    return [*WARMUPS, *PHRASES, *CONVERSATION]
