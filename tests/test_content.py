"""Coverage for cli/content.py — JSON-driven Parkinson's therapy program.

The current program.json is the LSVT LOUD-aligned version (v3.0).
Structure:
  warmup_ah          1 exercise  (L0_ah)         1 drill   (AH)
  pitch_glides       2 exercises (L0_high, L0_low) 2 drills (high AH, low AH)
  counting           1 exercise  (L0_count)      1 drill   (1..10)
  functional_phrases 2 exercises (L_func_basics, L_func_action) 10 drills
  sentences          2 exercises (L_sent_breath, L_sent_question) 7 drills
                                                  ---
                                                  21 drills total

If the program JSON is regenerated (different LSVT phrase set, custom
clinic content, etc.) update the constants at the top of this file —
not the assertion sites scattered through it.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import content
import pytest


# ---- Snapshot of the canonical program shape ----------------------------
# Update these alongside cli/program.json so the schema/structure tests
# continue to enforce the LSVT-aligned layout end-to-end.

CATEGORY_IDS = ["warmup_ah", "pitch_glides", "counting", "functional_phrases", "sentences"]
EXERCISES_BY_CATEGORY = {
    "warmup_ah":          ["L0_ah"],
    "pitch_glides":       ["L0_high", "L0_low"],
    "counting":           ["L0_count"],
    "functional_phrases": ["L_func_basics", "L_func_action"],
    "sentences":          ["L_sent_breath", "L_sent_question"],
}
DRILLS_PER_EXERCISE = {
    "L0_ah":          1,
    "L0_high":        1,
    "L0_low":         1,
    "L0_count":       1,
    "L_func_basics":  5,
    "L_func_action":  5,
    "L_sent_breath":  4,
    "L_sent_question": 3,
}
DRILLS_PER_CATEGORY = {
    "warmup_ah":          1,
    "pitch_glides":       2,
    "counting":           1,
    "functional_phrases": 10,
    "sentences":          7,
}
TOTAL_DRILLS = sum(DRILLS_PER_CATEGORY.values())  # 21
TOTAL_EXERCISES = sum(len(v) for v in EXERCISES_BY_CATEGORY.values())  # 8


# ---- Program loading -----------------------------------------------------

class TestProgramLoading:
    def test_loads_program_from_json(self):
        prog = content.load_program()
        assert prog.program_name.startswith("Parkinsons")
        # LSVT-aligned program is v3.x. Major version bumps signal a
        # content change worth re-reviewing the test snapshot above.
        assert prog.version.startswith("3.")
        assert prog.default_flow == ("main_task",)

    def test_program_has_all_canonical_categories(self):
        prog = content.load_program()
        ids = [c.id for c in prog.categories]
        assert ids == CATEGORY_IDS

    @pytest.mark.parametrize("category_id,expected_exercises", list(EXERCISES_BY_CATEGORY.items()))
    def test_category_has_expected_exercises(self, category_id, expected_exercises):
        cat = content.find_category(category_id)
        assert [e.id for e in cat.exercises] == expected_exercises

    def test_total_exercise_count(self):
        prog = content.load_program()
        total = sum(len(c.exercises) for c in prog.categories)
        assert total == TOTAL_EXERCISES

    def test_each_lesson_has_main_task_phase(self):
        prog = content.load_program()
        for cat in prog.categories:
            for ex in cat.exercises:
                phase_names = [p.name for p in ex.phases]
                assert phase_names == ["main_task"], \
                    f"{ex.id} has phases {phase_names}"

    def test_load_program_is_cached(self):
        a = content.load_program()
        b = content.load_program()
        assert a is b  # lru_cache

    def test_load_from_alternative_path(self, tmp_path):
        custom = tmp_path / "tiny.json"
        custom.write_text(json.dumps({
            "program_name": "tiny",
            "version": "0.1",
            "structure": {"default_flow": ["warmup", "main_task"]},
            "categories": [{
                "id": "x",
                "name": "X",
                "goal": "test",
                "exercises": [{
                    "id": "x_1",
                    "name": "Test",
                    "phases": {
                        "warmup": {"instructions": "breathe"},
                        "main_task": {"content": ["go"], "instructions": "say it"},
                    },
                }],
            }],
        }))
        # Bypass the lru_cache by passing path explicitly.
        content.load_program.cache_clear()
        try:
            prog = content.load_program(custom)
            assert prog.program_name == "tiny"
            assert len(prog.categories) == 1
            assert prog.categories[0].exercises[0].id == "x_1"
        finally:
            content.load_program.cache_clear()


# ---- Phase content normalisation ----------------------------------------

class TestContentCoercion:
    def test_string_content_becomes_one_tuple(self):
        assert content._coerce_content("1 to 10") == ("1 to 10",)

    def test_list_content_becomes_tuple(self):
        assert content._coerce_content(["a", "b"]) == ("a", "b")

    def test_missing_content_is_empty_tuple(self):
        assert content._coerce_content(None) == ()

    def test_non_string_in_list_is_stringified(self):
        assert content._coerce_content([1, 2.5]) == ("1", "2.5")

    def test_non_string_value_stringified(self):
        assert content._coerce_content(42) == ("42",)


# ---- Drill dataclass ----------------------------------------------------

class TestDrillDataclass:
    def test_is_frozen(self):
        d = content.Drill(stage="warmup", index=0, prompt="Ahhh")
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.prompt = "mutated"  # type: ignore[misc]

    def test_default_field_values(self):
        d = content.Drill(stage="warmup", index=0, prompt="Ahhh")
        assert d.note == ""
        assert d.target_dbfs == content.DEFAULT_TARGET_DBFS_FALLBACK
        assert d.category_id == ""
        assert d.exercise_name == ""
        assert d.focus == ""
        assert d.target_repetitions == 0
        assert d.target_duration_sec == 0


# ---- Flattening: exercise → drills --------------------------------------

class TestDrillsForExercise:
    def test_func_basics_unrolls_to_canonical_lsvt_phrases(self):
        """LSVT LOUD daily greetings: real-world phrases the patient
        actually says every day — NOT vowel sounds or fragments."""
        drills = content.drills_for_exercise("L_func_basics")
        assert [d.prompt for d in drills] == [
            "Good morning.",
            "How are you?",
            "I'm fine, thanks.",
            "Thank you.",
            "I love you.",
        ]
        assert all(d.stage == "main_task" for d in drills)

    def test_func_action_unrolls_to_canonical_lsvt_phrases(self):
        drills = content.drills_for_exercise("L_func_action")
        assert [d.prompt for d in drills] == [
            "I need help.",
            "Please sit down.",
            "Pass the food, please.",
            "Answer the phone.",
            "Shut the door.",
        ]

    def test_sustained_ah_is_one_drill(self):
        drills = content.drills_for_exercise("L0_ah")
        assert len(drills) == 1
        assert drills[0].prompt == "AH"
        # Foundational LSVT exercise — 6 reps, 6 seconds each.
        assert drills[0].target_repetitions == 6
        assert drills[0].target_duration_sec == 6

    def test_pitch_glide_high_drill_carries_repetitions(self):
        drills = content.drills_for_exercise("L0_high")
        assert len(drills) == 1
        assert drills[0].target_repetitions == 6
        assert drills[0].target_duration_sec == 5

    def test_count_one_to_ten_drill_present(self):
        drills = content.drills_for_exercise("L0_count")
        assert len(drills) == 1
        assert drills[0].prompt == "1 2 3 4 5 6 7 8 9 10"

    def test_question_sentences_drill(self):
        drills = content.drills_for_exercise("L_sent_question")
        assert [d.prompt for d in drills] == [
            "How are you today?",
            "Where are you going?",
            "Would you like to go out to eat?",
        ]
        # Each prompt is a real question — none of them are single
        # letters or vowel fragments.
        for d in drills:
            assert d.prompt.endswith("?")
            assert len(d.prompt.split()) >= 3, f"too short: {d.prompt!r}"

    def test_main_task_drill_carries_focus(self):
        drills = content.drills_for_exercise("L_func_basics")
        first = drills[0]
        assert "loud" in first.focus.lower() or "project" in first.focus.lower()
        assert first.target_repetitions == 1

    def test_instructions_become_note(self):
        drills = content.drills_for_exercise("L_func_basics")
        # When a phase has explicit content, instructions become the note.
        assert "loud" in drills[0].note.lower()

    def test_each_drill_has_category_and_exercise_metadata(self):
        for d in content.drills_for_exercise("L_func_basics"):
            assert d.category_id == "functional_phrases"
            assert d.category_name.startswith("Functional Phrases")
            assert d.exercise_id == "L_func_basics"
            assert d.exercise_name == "Daily Greetings & Replies"

    def test_target_dbfs_uses_main_task_level(self):
        drills = content.drills_for_exercise("L_func_basics")
        for d in drills:
            assert d.target_dbfs == content.DEFAULT_TARGET_DBFS_BY_STAGE["main_task"]

    def test_unknown_exercise_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_exercise("nope_99")


# ---- drills_for_category ------------------------------------------------

class TestDrillsForCategory:
    @pytest.mark.parametrize("category_id,expected_count", list(DRILLS_PER_CATEGORY.items()))
    def test_category_drill_count(self, category_id, expected_count):
        drills = content.drills_for_category(category_id)
        assert len(drills) == expected_count

    def test_unknown_category_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_category("nope")


# ---- default_drill_set --------------------------------------------------

class TestDefaultDrillSet:
    def test_default_walks_entire_program(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        assert len(drills) == TOTAL_DRILLS
        # All canonical categories appear.
        assert {d.category_id for d in drills} == set(CATEGORY_IDS)
        # All exercises appear.
        assert len({d.exercise_id for d in drills}) == TOTAL_EXERCISES

    def test_default_drills_are_in_program_order(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        # Categories appear in JSON-declared order, no interleaving.
        seen: list[str] = []
        for d in drills:
            if not seen or seen[-1] != d.category_id:
                seen.append(d.category_id)
        assert seen == CATEGORY_IDS

    def test_exercise_override_beats_category_and_default(self, monkeypatch):
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "functional_phrases")
        monkeypatch.setenv("VOICE_COACH_EXERCISE", "L_func_action")
        drills = content.default_drill_set()
        assert all(d.exercise_id == "L_func_action" for d in drills)
        assert len(drills) == DRILLS_PER_EXERCISE["L_func_action"]

    def test_category_override_returns_only_that_category(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "functional_phrases")
        drills = content.default_drill_set()
        assert all(d.category_id == "functional_phrases" for d in drills)
        assert len(drills) == DRILLS_PER_CATEGORY["functional_phrases"]

    def test_unknown_exercise_override_raises(self, monkeypatch):
        monkeypatch.setenv("VOICE_COACH_EXERCISE", "nonexistent")
        with pytest.raises(KeyError):
            content.default_drill_set()

    def test_unknown_category_override_raises(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "nope")
        with pytest.raises(KeyError):
            content.default_drill_set()


class TestAllDrills:
    def test_all_drills_returns_full_program(self):
        drills = content.all_drills()
        assert len(drills) == TOTAL_DRILLS

    def test_all_drills_per_category_count(self):
        drills = content.all_drills()
        per_cat: dict[str, int] = {}
        for d in drills:
            per_cat[d.category_id] = per_cat.get(d.category_id, 0) + 1
        assert per_cat == DRILLS_PER_CATEGORY

    def test_all_drills_per_exercise_count(self):
        drills = content.all_drills()
        per_ex: dict[str, int] = {}
        for d in drills:
            per_ex[d.exercise_id] = per_ex.get(d.exercise_id, 0) + 1
        assert per_ex == DRILLS_PER_EXERCISE

    def test_all_drills_first_is_warmup_ah(self):
        """Program order starts with the LSVT foundation: sustained AH."""
        drills = content.all_drills()
        first = drills[0]
        assert first.stage == "main_task"
        assert first.exercise_id == "L0_ah"
        assert first.category_id == "warmup_ah"
        assert first.prompt == "AH"

    def test_all_drills_last_is_a_sentence_question(self):
        drills = content.all_drills()
        last = drills[-1]
        assert last.stage == "main_task"
        assert last.category_id == "sentences"
        # The final question in the last sentences exercise.
        assert last.prompt.endswith("?")

    def test_every_drill_has_exercise_metadata(self):
        for d in content.all_drills():
            assert d.exercise_id, f"drill missing exercise_id: {d}"
            assert d.exercise_name, f"drill missing exercise_name: {d}"
            assert d.category_id, f"drill missing category_id: {d}"
            assert d.category_name, f"drill missing category_name: {d}"


class TestEmptyProgram:
    def test_empty_program_resolves_to_empty_drill_set(self, tmp_path, monkeypatch):
        """Defensive: a program with no categories yields []."""
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({
            "program_name": "empty",
            "version": "0.0",
            "structure": {"default_flow": []},
            "categories": [],
        }))
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        content.load_program.cache_clear()
        original_path = content.PROGRAM_JSON_PATH
        try:
            monkeypatch.setattr(content, "PROGRAM_JSON_PATH", empty)
            assert content.all_drills() == []
            assert content.default_drill_set() == []
        finally:
            monkeypatch.setattr(content, "PROGRAM_JSON_PATH", original_path)
            content.load_program.cache_clear()


# ---- all_exercises + find_exercise --------------------------------------

class TestQueryHelpers:
    def test_all_exercises_returns_all_pairs(self):
        pairs = content.all_exercises()
        assert len(pairs) == TOTAL_EXERCISES
        for cat, ex in pairs:
            assert isinstance(cat, content.Category)
            assert isinstance(ex, content.Exercise)

    def test_find_exercise_returns_correct_pair(self):
        cat, ex = content.find_exercise("L_sent_question")
        assert cat.id == "sentences"
        assert ex.name == "Ask Questions Clearly"

    def test_find_exercise_unknown_raises(self):
        with pytest.raises(KeyError):
            content.find_exercise("nope")

    def test_find_category_unknown_raises(self):
        with pytest.raises(KeyError):
            content.find_category("nope")


# ---- Invariants across the whole program -------------------------------

class TestProgramInvariants:
    def test_every_exercise_id_unique(self):
        ids = [ex.id for _, ex in content.all_exercises()]
        assert len(ids) == len(set(ids)), f"duplicate exercise ids: {ids}"

    def test_every_drill_has_valid_stage(self):
        for _, ex in content.all_exercises():
            for d in content.drills_for_exercise(ex.id):
                assert d.stage in content.VALID_STAGES, \
                    f"invalid stage {d.stage!r} on {d}"

    def test_every_drill_has_nonempty_prompt(self):
        for _, ex in content.all_exercises():
            for d in content.drills_for_exercise(ex.id):
                assert d.prompt.strip(), f"empty prompt: {d}"

    def test_target_dbfs_is_reasonable(self):
        for _, ex in content.all_exercises():
            for d in content.drills_for_exercise(ex.id):
                assert -40.0 <= d.target_dbfs <= 0.0
