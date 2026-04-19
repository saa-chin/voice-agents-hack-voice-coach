"""Coverage for cli/content.py — JSON-driven Parkinson's therapy program."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import content
import pytest


# ---- Program loading -----------------------------------------------------

class TestProgramLoading:
    def test_loads_program_from_json(self):
        prog = content.load_program()
        assert prog.program_name.startswith("Parkinsons")
        assert prog.version == "2.0"
        assert prog.default_flow == ("main_task",)

    def test_program_has_all_three_categories(self):
        prog = content.load_program()
        ids = [c.id for c in prog.categories]
        assert ids == ["words", "names", "sentences"]

    def test_words_has_three_lessons(self):
        cat = content.find_category("words")
        assert len(cat.exercises) == 3
        assert [e.id for e in cat.exercises] == ["L1", "L3", "L7"]

    def test_names_has_two_lessons(self):
        cat = content.find_category("names")
        assert [e.id for e in cat.exercises] == ["L2", "L6"]

    def test_sentences_has_five_lessons(self):
        cat = content.find_category("sentences")
        assert [e.id for e in cat.exercises] == ["L4", "L5", "L8", "L9", "L10"]

    def test_total_lesson_count(self):
        prog = content.load_program()
        total = sum(len(c.exercises) for c in prog.categories)
        # 3 + 2 + 5 = 10
        assert total == 10

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
    def test_L1_unrolls_to_five_drills(self):
        drills = content.drills_for_exercise("L1")
        # main_task with 5 content items (prompt + 4 variations) = 5
        assert len(drills) == 5
        assert all(d.stage == "main_task" for d in drills)
        assert [d.prompt for d in drills] == [
            "New York", "London", "Paris", "Tokyo", "Sydney",
        ]

    def test_L3_unrolls_everyday_action_words(self):
        drills = content.drills_for_exercise("L3")
        assert [d.prompt for d in drills] == [
            "Help", "Stop", "Go", "Wait", "Come",
        ]

    def test_L7_has_three_drills(self):
        drills = content.drills_for_exercise("L7")
        assert [d.prompt for d in drills] == [
            "California", "Mississippi", "Philadelphia",
        ]

    def test_L10_prompt_response_has_single_drill(self):
        drills = content.drills_for_exercise("L10")
        assert len(drills) == 1
        assert drills[0].prompt == "What did you eat today?"

    def test_main_task_drill_carries_focus(self):
        drills = content.drills_for_exercise("L1")
        main = drills[0]
        assert "clear consonants" in main.focus.lower()
        assert main.target_repetitions == 1

    def test_instructions_become_note(self):
        drills = content.drills_for_exercise("L1")
        # When a phase has explicit content, instructions become the note.
        assert drills[0].note.startswith("Say each city name clearly")

    def test_each_drill_has_category_and_exercise_metadata(self):
        for d in content.drills_for_exercise("L1"):
            assert d.category_id == "words"
            assert d.category_name == "Words"
            assert d.exercise_id == "L1"
            assert d.exercise_name == "Speak Strong City Names"

    def test_target_dbfs_uses_main_task_level(self):
        drills = content.drills_for_exercise("L1")
        for d in drills:
            assert d.target_dbfs == content.DEFAULT_TARGET_DBFS_BY_STAGE["main_task"]

    def test_unknown_exercise_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_exercise("nope_99")


# ---- drills_for_category ------------------------------------------------

class TestDrillsForCategory:
    def test_words_total_drills(self):
        drills = content.drills_for_category("words")
        # L1: 5, L3: 5, L7: 3 → 13
        assert len(drills) == 13

    def test_names_total_drills(self):
        drills = content.drills_for_category("names")
        # L2: 5, L6: 4 → 9
        assert len(drills) == 9

    def test_sentences_total_drills(self):
        drills = content.drills_for_category("sentences")
        # L4: 4, L5: 3, L8: 3, L9: 3, L10: 1 → 14
        assert len(drills) == 14

    def test_names_L6_prompts_include_exclamation(self):
        drills = content.drills_for_category("names")
        l6_prompts = [d.prompt for d in drills if d.exercise_id == "L6"]
        assert l6_prompts == ["Anna!", "Tom!", "Lisa!", "Mark!"]

    def test_unknown_category_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_category("nope")


# ---- default_drill_set --------------------------------------------------

class TestDefaultDrillSet:
    def test_default_walks_entire_program(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        # Full program: 13 + 9 + 14 = 36
        assert len(drills) == 36
        # All three categories appear.
        cats = {d.category_id for d in drills}
        assert cats == {"words", "names", "sentences"}
        # All 10 lessons appear.
        assert len({d.exercise_id for d in drills}) == 10

    def test_default_drills_are_in_program_order(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        # Categories appear in JSON-declared order, no interleaving.
        seen: list[str] = []
        for d in drills:
            if not seen or seen[-1] != d.category_id:
                seen.append(d.category_id)
        assert seen == ["words", "names", "sentences"]

    def test_exercise_override_beats_category_and_default(self, monkeypatch):
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "words")
        monkeypatch.setenv("VOICE_COACH_EXERCISE", "L6")
        drills = content.default_drill_set()
        # Lesson override wins.
        assert all(d.exercise_id == "L6" for d in drills)
        assert len(drills) == 4

    def test_category_override_returns_only_that_category(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "names")
        drills = content.default_drill_set()
        assert all(d.category_id == "names" for d in drills)
        # L2: 5, L6: 4 → 9
        assert len(drills) == 9

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
        assert len(drills) == 36

    def test_all_drills_per_category_count(self):
        drills = content.all_drills()
        per_cat: dict[str, int] = {}
        for d in drills:
            per_cat[d.category_id] = per_cat.get(d.category_id, 0) + 1
        assert per_cat == {
            "words": 13,
            "names": 9,
            "sentences": 14,
        }

    def test_all_drills_per_exercise_count(self):
        drills = content.all_drills()
        per_ex: dict[str, int] = {}
        for d in drills:
            per_ex[d.exercise_id] = per_ex.get(d.exercise_id, 0) + 1
        assert per_ex == {
            "L1": 5, "L3": 5, "L7": 3,
            "L2": 5, "L6": 4,
            "L4": 4, "L5": 3, "L8": 3, "L9": 3, "L10": 1,
        }

    def test_all_drills_first_is_L1_new_york(self):
        drills = content.all_drills()
        first = drills[0]
        assert first.stage == "main_task"
        assert first.exercise_id == "L1"
        assert first.category_id == "words"
        assert first.prompt == "New York"

    def test_all_drills_last_is_L10_prompt(self):
        drills = content.all_drills()
        last = drills[-1]
        assert last.stage == "main_task"
        assert last.exercise_id == "L10"
        assert last.prompt == "What did you eat today?"

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
        assert len(pairs) == 10
        for cat, ex in pairs:
            assert isinstance(cat, content.Category)
            assert isinstance(ex, content.Exercise)

    def test_find_exercise_returns_correct_pair(self):
        cat, ex = content.find_exercise("L8")
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
