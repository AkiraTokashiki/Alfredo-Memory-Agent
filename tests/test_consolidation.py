"""Tests for memory consolidation and stale-memory archival."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from memory_agent.core.config import ConsolidationConfig
from memory_agent.core.consolidation import ConsolidationAction, MemoryConsolidator

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


@pytest.fixture
def store() -> MemoryStore:
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    s = MemoryStore(db.name)
    s.initialize()
    yield s
    s.close()
    Path(db.name).unlink(missing_ok=True)


def test_archive_memory_records_reason(store: MemoryStore):
    memory_id = store.add_memory(
        MemoryRecord(content="old preference", metadata={"topic": "language"})
    )

    store.archive_memory(memory_id, reason="superseded", metadata={"superseded_by": 99})

    archived = store.get_memory(memory_id)
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "superseded"
    assert archived.metadata["superseded_by"] == 99


def test_archive_below_threshold_records_decay_reason(store: MemoryStore):
    memory_id = store.add_memory(MemoryRecord(content="weak", strength=0.01))

    count = store.archive_below_threshold(0.05)

    archived = store.get_memory(memory_id)
    assert count == 1
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "decay"



class FakeSimilarity:
    def __init__(self, scores: dict[tuple[str, str], float] | None = None) -> None:
        self.scores = scores or {}

    def similarity(self, left: str, right: str) -> float:
        return self.scores.get((left, right), self.scores.get((right, left), 0.0))


def test_duplicate_preference_reinforces_existing(store: MemoryStore):
    existing = MemoryRecord(
        content="El usuario prefiere: Python",
        memory_type="preference",
        strength=0.4,
    )
    existing_id = store.add_memory(existing)
    candidate = MemoryRecord(
        content="El usuario prefiere: programar en Python",
        memory_type="preference",
    )
    similarity = FakeSimilarity({(existing.content, candidate.content): 0.93})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    decision = consolidator.consolidate(candidate)

    refreshed = store.get_memory(existing_id)
    assert decision.action is ConsolidationAction.REINFORCE
    assert decision.existing_memory_id == existing_id
    assert refreshed is not None
    assert refreshed.strength > 0.4
    assert store.count_memories() == 1


def test_conflicting_preference_supersedes_existing(store: MemoryStore):
    old = MemoryRecord(
        content="El usuario prefiere: Python",
        memory_type="preference",
        metadata={"topic": "python", "polarity": "positive"},
    )
    old_id = store.add_memory(old)
    new = MemoryRecord(
        content="Al usuario no le gusta: Python",
        memory_type="preference",
        metadata={"topic": "python", "polarity": "negative"},
    )
    similarity = FakeSimilarity({(old.content, new.content): 0.70})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    decision = consolidator.consolidate(new)

    archived_old = store.get_memory(old_id)
    stored_new = store.get_memory(decision.new_memory_id or -1)
    assert decision.action is ConsolidationAction.UPDATE
    assert archived_old is not None
    assert archived_old.is_active is False
    assert archived_old.metadata["archival_reason"] == "superseded"
    assert stored_new is not None
    assert stored_new.is_active is True
    assert stored_new.metadata["supersedes"] == old_id


def test_explicit_forget_archives_matching_memory(store: MemoryStore):
    memory_id = store.add_memory(
        MemoryRecord(content="El usuario prefiere: Python", memory_type="preference")
    )
    similarity = FakeSimilarity({("forget python", "El usuario prefiere: Python"): 0.8})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    archived_count = consolidator.forget_matching("forget python")

    archived = store.get_memory(memory_id)
    assert archived_count == 1
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "explicit_user_request"