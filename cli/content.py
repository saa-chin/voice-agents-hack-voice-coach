"""Drill content for the voice-coach CLI.

The clinical program (categories → exercises → phases) is loaded from
`cli/program.json` — the canonical, single source of truth shared by the
CLI, the FastAPI server, and (manually mirrored) the mobile app.

Public API (kept stable for downstream callers):
  - Drill                  dataclass — one step in the coach loop
  - default_drill_set()    -> list[Drill]    flattened drills for the
                                              default exercise (the first
                                              exercise of the first category,
                                              i.e. vl_1 "Sustained Vowel Power")
  - drills_for_exercise(exercise_id)
  - drills_for_category(category_id)
  - load_program()         -> Program        full structured program
  - default_exercise_id    str               configurable via env

Environment overrides:
  - VOICE_COACH_EXERCISE   pick a specific exercise id (e.g. "ar_1")
  - VOICE_COACH_PROGRAM    path to a custom program JSON file

`target_dbfs` is a relative loudness target (dBFS), NOT a calibrated
dB-SPL value. Calibrating mic input to room SPL needs a reference meter
(R03 in the implementation doc); until that lands we report dBFS and
label it as such.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# --- Constants -----------------------------------------------------------

VALID_STAGES: tuple[str, ...] = ("warmup", "glide", "counting", "main_task")

# A target loudness in dBFS. The coach prompts Gemma to grade against this.
# Phases vary in intensity:
#   warmup / glide  → soft, sustained → low target
#   counting        → spoken normally → mid target
#   main_task       → "loud and clear" → louder target
DEFAULT_TARGET_DBFS_BY_STAGE: dict[str, float] = {
    "warmup": -22.0,
    "glide": -22.0,
    "counting": -18.0,
    "main_task": -15.0,
}

DEFAULT_TARGET_DBFS_FALLBACK = -18.0

PROGRAM_JSON_PATH = Path(
    os.environ.get(
        "VOICE_COACH_PROGRAM",
        str(Path(__file__).resolve().parent / "program.json"),
    )
)


# --- Schema dataclasses --------------------------------------------------

@dataclass(frozen=True)
class Phase:
    name: str                       # one of VALID_STAGES
    instructions: str
    content: tuple[str, ...] = ()   # 0+ explicit content items
    focus: str = ""
    repetitions: int = 1
    duration_sec: int = 0


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    phases: tuple[Phase, ...] = ()


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    goal: str
    exercises: tuple[Exercise, ...] = ()


@dataclass(frozen=True)
class Program:
    program_name: str
    version: str
    default_flow: tuple[str, ...]
    categories: tuple[Category, ...]


@dataclass(frozen=True)
class Drill:
    """One step the coach takes the user through.

    Backward-compatible with the previous schema (`stage`, `index`, `prompt`,
    `note`, `target_dbfs`); all new fields are keyword-only with sensible
    defaults so older callers / tests still work.
    """
    stage: str            # one of VALID_STAGES
    index: int            # position within stage, per exercise
    prompt: str           # what the user should say (or the instruction itself
                          # for instruction-only phases like warmup/glide)
    note: str = ""        # the JSON `instructions` text (a coaching cue)
    target_dbfs: float = DEFAULT_TARGET_DBFS_FALLBACK

    # Richer fields exposed to the model + UI:
    category_id: str = ""
    category_name: str = ""
    exercise_id: str = ""
    exercise_name: str = ""
    focus: str = ""
    target_repetitions: int = 0
    target_duration_sec: int = 0


# --- Loader --------------------------------------------------------------

def _coerce_content(raw: Any) -> tuple[str, ...]:
    """Normalize the JSON `content` field (str | list[str] | missing) to a tuple."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    return (str(raw),)


def _build_phase(name: str, raw: dict[str, Any]) -> Phase:
    return Phase(
        name=name,
        instructions=str(raw.get("instructions", "")).strip(),
        content=_coerce_content(raw.get("content")),
        focus=str(raw.get("focus", "")).strip(),
        repetitions=int(raw.get("repetitions", 1)),
        duration_sec=int(raw.get("duration_sec", 0)),
    )


def _build_exercise(raw: dict[str, Any], default_flow: tuple[str, ...]) -> Exercise:
    phases_dict = raw.get("phases", {}) or {}
    # Preserve the program's declared default_flow ordering.
    phases = tuple(
        _build_phase(name, phases_dict[name])
        for name in default_flow
        if name in phases_dict
    )
    return Exercise(
        id=str(raw["id"]),
        name=str(raw.get("name", raw["id"])),
        phases=phases,
    )


def _build_program(raw: dict[str, Any]) -> Program:
    default_flow = tuple(
        raw.get("structure", {}).get("default_flow", VALID_STAGES)
    )
    categories = tuple(
        Category(
            id=str(c["id"]),
            name=str(c.get("name", c["id"])),
            goal=str(c.get("goal", "")),
            exercises=tuple(
                _build_exercise(e, default_flow)
                for e in c.get("exercises", [])
            ),
        )
        for c in raw.get("categories", [])
    )
    return Program(
        program_name=str(raw.get("program_name", "Voice Coach Program")),
        version=str(raw.get("version", "0.0")),
        default_flow=default_flow,
        categories=categories,
    )


@lru_cache(maxsize=1)
def load_program(path: Path | None = None) -> Program:
    """Load + parse the program JSON. Cached after first call."""
    p = path or PROGRAM_JSON_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return _build_program(raw)


# --- Default-exercise selection -----------------------------------------

def _resolve_default_exercise_id() -> str:
    explicit = os.environ.get("VOICE_COACH_EXERCISE", "").strip()
    if explicit:
        return explicit
    prog = load_program()
    if prog.categories and prog.categories[0].exercises:
        return prog.categories[0].exercises[0].id
    return ""


# --- Flattening: Exercise → list[Drill] ---------------------------------

def _phase_prompt_pairs(phase: Phase) -> list[tuple[str, str]]:
    """Yield (prompt, note) per drill step within a phase.

    - Phases with explicit `content` produce one drill per content item;
      prompt = item, note = instructions.
    - Phases without `content` (warmup, glide) produce one drill;
      prompt = instructions, note = "" (the instruction IS the cue).
    """
    if not phase.content:
        return [(phase.instructions, "")]
    return [(item, phase.instructions) for item in phase.content]


def _drills_from_exercise(
    exercise: Exercise,
    category: Category,
) -> list[Drill]:
    """Flatten one exercise into its sequence of Drill steps."""
    out: list[Drill] = []
    for phase in exercise.phases:
        pairs = _phase_prompt_pairs(phase)
        target = DEFAULT_TARGET_DBFS_BY_STAGE.get(
            phase.name, DEFAULT_TARGET_DBFS_FALLBACK
        )
        for i, (prompt, note) in enumerate(pairs):
            out.append(Drill(
                stage=phase.name,
                index=i,
                prompt=prompt,
                note=note,
                target_dbfs=target,
                category_id=category.id,
                category_name=category.name,
                exercise_id=exercise.id,
                exercise_name=exercise.name,
                focus=phase.focus,
                target_repetitions=phase.repetitions,
                target_duration_sec=phase.duration_sec,
            ))
    return out


# --- Public API ---------------------------------------------------------

def all_exercises() -> list[tuple[Category, Exercise]]:
    prog = load_program()
    return [(c, e) for c in prog.categories for e in c.exercises]


def find_exercise(exercise_id: str) -> tuple[Category, Exercise]:
    for cat, ex in all_exercises():
        if ex.id == exercise_id:
            return cat, ex
    raise KeyError(f"unknown exercise id: {exercise_id!r}")


def find_category(category_id: str) -> Category:
    prog = load_program()
    for c in prog.categories:
        if c.id == category_id:
            return c
    raise KeyError(f"unknown category id: {category_id!r}")


def drills_for_exercise(exercise_id: str) -> list[Drill]:
    cat, ex = find_exercise(exercise_id)
    return _drills_from_exercise(ex, cat)


def drills_for_category(category_id: str) -> list[Drill]:
    cat = find_category(category_id)
    out: list[Drill] = []
    for ex in cat.exercises:
        out.extend(_drills_from_exercise(ex, cat))
    return out


def default_drill_set() -> list[Drill]:
    """Default sequence: one exercise (configurable via VOICE_COACH_EXERCISE).

    Defaults to the first exercise of the first category in the program
    (vl_1 "Sustained Vowel Power" — 4 drill steps walking the full
    warmup → glide → counting → main_task flow).
    """
    eid = _resolve_default_exercise_id()
    if not eid:
        return []
    return drills_for_exercise(eid)
