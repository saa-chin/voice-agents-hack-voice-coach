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
        assert prog.version == "1.0"
        assert prog.default_flow == ("warmup", "glide", "counting", "main_task")

    def test_program_has_all_four_categories(self):
        prog = content.load_program()
        ids = [c.id for c in prog.categories]
        assert ids == ["voice_loudness", "prosody", "articulation", "functional"]

    def test_voice_loudness_has_five_exercises(self):
        cat = content.find_category("voice_loudness")
        assert len(cat.exercises) == 5
        assert [e.id for e in cat.exercises] == ["vl_1", "vl_2", "vl_3", "vl_4", "vl_5"]

    def test_total_exercise_count(self):
        prog = content.load_program()
        total = sum(len(c.exercises) for c in prog.categories)
        # 5 + 2 + 2 + 3 = 12
        assert total == 12

    def test_each_exercise_has_all_default_phases(self):
        prog = content.load_program()
        for cat in prog.categories:
            for ex in cat.exercises:
                phase_names = [p.name for p in ex.phases]
                assert phase_names == list(prog.default_flow), \
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
    def test_vl_1_flattens_to_four_drills(self):
        drills = content.drills_for_exercise("vl_1")
        # warmup + glide + counting + main_task (1 content item) = 4
        assert len(drills) == 4
        stages = [d.stage for d in drills]
        assert stages == ["warmup", "glide", "counting", "main_task"]

    def test_vl_2_unrolls_main_task_content(self):
        drills = content.drills_for_exercise("vl_2")
        # warmup + glide + counting + 5 main_task content items = 8
        assert len(drills) == 8
        main_tasks = [d for d in drills if d.stage == "main_task"]
        assert [d.prompt for d in main_tasks] == ["Hey", "Stop", "Hello", "Yes", "No"]

    def test_warmup_drill_uses_instruction_as_prompt(self):
        drills = content.drills_for_exercise("vl_1")
        warmup = drills[0]
        assert warmup.stage == "warmup"
        # No explicit content → instruction text becomes the prompt.
        assert "deep diaphragmatic breath" in warmup.prompt
        assert warmup.note == ""

    def test_main_task_drill_carries_focus(self):
        drills = content.drills_for_exercise("vl_1")
        main = next(d for d in drills if d.stage == "main_task")
        assert main.focus == "Breath control and vocal strength"
        assert main.target_repetitions == 10
        assert main.target_duration_sec == 5

    def test_counting_uses_content_as_prompt_with_instruction_as_note(self):
        drills = content.drills_for_exercise("vl_1")
        counting = next(d for d in drills if d.stage == "counting")
        assert counting.prompt == "1 to 10"
        assert counting.note == "Count loudly and clearly"

    def test_each_drill_has_category_and_exercise_metadata(self):
        for d in content.drills_for_exercise("vl_1"):
            assert d.category_id == "voice_loudness"
            assert d.category_name == "Voice Loudness & Breath Support"
            assert d.exercise_id == "vl_1"
            assert d.exercise_name == "Sustained Vowel Power"

    def test_target_dbfs_varies_by_stage(self):
        drills = content.drills_for_exercise("vl_1")
        by_stage = {d.stage: d.target_dbfs for d in drills}
        # main_task should be louder (higher dBFS, less negative) than warmup.
        assert by_stage["main_task"] > by_stage["warmup"]

    def test_unknown_exercise_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_exercise("nope_99")


# ---- drills_for_category ------------------------------------------------

class TestDrillsForCategory:
    def test_voice_loudness_total_drills(self):
        drills = content.drills_for_category("voice_loudness")
        # vl_1: 4, vl_2: 8, vl_3: 6, vl_4: 6, vl_5: 4 → 28
        assert len(drills) == 28

    def test_articulation_pa_ta_ka_unrolls_correctly(self):
        drills = content.drills_for_category("articulation")
        ar1_main = [d for d in drills if d.exercise_id == "ar_1" and d.stage == "main_task"]
        assert len(ar1_main) == 1
        assert ar1_main[0].prompt == "Pa-Ta-Ka"
        assert ar1_main[0].focus == "Tongue and lip coordination"

    def test_functional_word_reading_unrolls_5_words(self):
        drills = content.drills_for_category("functional")
        fn1_main = [d for d in drills if d.exercise_id == "fn_1" and d.stage == "main_task"]
        assert [d.prompt for d in fn1_main] == ["cat", "dog", "sun", "book", "chair"]

    def test_unknown_category_raises_keyerror(self):
        with pytest.raises(KeyError):
            content.drills_for_category("nope")


# ---- default_drill_set --------------------------------------------------

class TestDefaultDrillSet:
    def test_default_walks_entire_program(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        # Full clinical program: 28 + 11 + 9 + 21 = 69
        assert len(drills) == 69
        # All four categories appear.
        cats = {d.category_id for d in drills}
        assert cats == {"voice_loudness", "prosody", "articulation", "functional"}
        # All 12 exercises appear.
        assert len({d.exercise_id for d in drills}) == 12

    def test_default_drills_are_in_program_order(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.delenv("VOICE_COACH_CATEGORY", raising=False)
        drills = content.default_drill_set()
        # Categories appear in JSON-declared order, no interleaving.
        seen: list[str] = []
        for d in drills:
            if not seen or seen[-1] != d.category_id:
                seen.append(d.category_id)
        assert seen == ["voice_loudness", "prosody", "articulation", "functional"]

    def test_exercise_override_beats_category_and_default(self, monkeypatch):
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "voice_loudness")
        monkeypatch.setenv("VOICE_COACH_EXERCISE", "ar_1")
        drills = content.default_drill_set()
        # Exercise override wins.
        assert all(d.exercise_id == "ar_1" for d in drills)
        assert len(drills) == 4

    def test_category_override_returns_only_that_category(self, monkeypatch):
        monkeypatch.delenv("VOICE_COACH_EXERCISE", raising=False)
        monkeypatch.setenv("VOICE_COACH_CATEGORY", "articulation")
        drills = content.default_drill_set()
        assert all(d.category_id == "articulation" for d in drills)
        # ar_1: 4, ar_2: 5 → 9
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
        assert len(drills) == 69

    def test_all_drills_per_category_count(self):
        drills = content.all_drills()
        per_cat: dict[str, int] = {}
        for d in drills:
            per_cat[d.category_id] = per_cat.get(d.category_id, 0) + 1
        assert per_cat == {
            "voice_loudness": 28,
            "prosody": 11,
            "articulation": 9,
            "functional": 21,
        }

    def test_all_drills_per_exercise_count(self):
        drills = content.all_drills()
        per_ex: dict[str, int] = {}
        for d in drills:
            per_ex[d.exercise_id] = per_ex.get(d.exercise_id, 0) + 1
        assert per_ex == {
            "vl_1": 4, "vl_2": 8, "vl_3": 6, "vl_4": 6, "vl_5": 4,
            "pr_1": 5, "pr_2": 6,
            "ar_1": 4, "ar_2": 5,
            "fn_1": 8, "fn_2": 6, "fn_3": 7,
        }

    def test_all_drills_first_is_warmup_of_vl_1(self):
        drills = content.all_drills()
        first = drills[0]
        assert first.stage == "warmup"
        assert first.exercise_id == "vl_1"
        assert first.category_id == "voice_loudness"

    def test_all_drills_last_is_main_task_of_fn_3(self):
        drills = content.all_drills()
        last = drills[-1]
        assert last.stage == "main_task"
        assert last.exercise_id == "fn_3"
        assert last.prompt == "PHONE"

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
        assert len(pairs) == 12
        for cat, ex in pairs:
            assert isinstance(cat, content.Category)
            assert isinstance(ex, content.Exercise)

    def test_find_exercise_returns_correct_pair(self):
        cat, ex = content.find_exercise("pr_2")
        assert cat.id == "prosody"
        assert ex.name == "Emotion Expression"

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
