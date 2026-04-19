"""Coverage for cli/content.py — pure data, but the invariants matter."""
from __future__ import annotations

import dataclasses

import content
import pytest


def test_drill_is_frozen_dataclass():
    d = content.Drill("warmup", 0, "Aaaah")
    assert dataclasses.is_dataclass(d)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.prompt = "mutated"  # type: ignore[misc]


def test_drill_default_target_dbfs():
    d = content.Drill("phrase", 1, "Hello")
    assert d.target_dbfs == -18.0
    assert d.note == ""


def test_warmups_phrases_conversation_nonempty():
    assert len(content.WARMUPS) >= 3
    assert len(content.PHRASES) >= 5
    assert len(content.CONVERSATION) >= 2


def test_default_drill_set_order_warmup_phrase_conversation():
    drills = content.default_drill_set()
    stages = [d.stage for d in drills]
    # stages should appear in this order with no interleaving
    expected_order = ["warmup", "phrase", "conversation"]
    seen: list[str] = []
    for s in stages:
        if not seen or seen[-1] != s:
            seen.append(s)
    assert seen == expected_order, f"unexpected stage ordering: {seen}"


def test_default_drill_set_total_count():
    drills = content.default_drill_set()
    assert len(drills) == len(content.WARMUPS) + len(content.PHRASES) + len(content.CONVERSATION)


def test_every_drill_has_valid_stage():
    valid_stages = {"warmup", "phrase", "conversation"}
    for d in content.default_drill_set():
        assert d.stage in valid_stages, f"invalid stage: {d.stage!r}"


def test_every_drill_has_nonempty_prompt():
    for d in content.default_drill_set():
        assert d.prompt.strip(), f"empty prompt for {d!r}"


def test_indexes_are_zero_based_within_stage():
    by_stage: dict[str, list[content.Drill]] = {}
    for d in content.default_drill_set():
        by_stage.setdefault(d.stage, []).append(d)
    for stage, drills in by_stage.items():
        indexes = [d.index for d in drills]
        assert indexes == list(range(len(drills))), \
            f"{stage} indexes not 0-based: {indexes}"


def test_target_dbfs_is_reasonable():
    """dBFS targets must be negative (full-scale digital is 0 dBFS)
    and not so quiet they're meaningless."""
    for d in content.default_drill_set():
        assert -40.0 <= d.target_dbfs <= 0.0, f"bad target {d.target_dbfs} on {d.prompt!r}"


def test_warmups_have_notes():
    """Warm-up vowels need coaching notes for clinical clarity."""
    for d in content.WARMUPS:
        assert d.note, f"warmup {d.prompt!r} missing note"
