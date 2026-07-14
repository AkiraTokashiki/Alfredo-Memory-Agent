"""RED contracts for deterministic, idempotent episodic consolidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_agent.core.episodes import (
    EpisodeSummary,
    EpisodeSummaryBuilder,
    consolidate_session,
)
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


@pytest.fixture
def memory_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "episodes.db")
    store.initialize()
    try:
        yield store
    finally:
        store.close()


def _events() -> list[dict[str, object]]:
    return [
        {
            "turn_index": 1,
            "role": "user",
            "content": "We diagnosed the import failure and found a missing dependency.",
        },
        {
            "turn_index": 2,
            "role": "assistant",
            "content": "The dependency was added and the focused test now passes.",
        },
        {
            "turn_index": 3,
            "role": "user",
            "content": "Record the fix so the next release can reuse it.",
        },
    ]


def test_episode_summary_builder_is_deterministic_and_json_safe() -> None:
    events = _events()
    builder = EpisodeSummaryBuilder()

    first = builder.build(events, session_id=17, namespace="tenant-a")
    second = builder.build(events, session_id=17, namespace="tenant-a")

    assert isinstance(first, EpisodeSummary)
    assert first.to_dict() == second.to_dict()
    assert first.session_id == 17
    assert first.namespace == "tenant-a"
    assert first.summary
    assert "missing dependency" in first.summary
    assert "focused test" in first.summary


def test_consolidate_session_persists_one_summary_across_reopen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "reopen-episodes.db"
    store = MemoryStore(db_path)
    store.initialize()
    session_id = store.create_session("dependency-fix", namespace="tenant-a")

    for event in _events():
        memory_id = store.add_memory(
            MemoryRecord(
                content=str(event["content"]),
                memory_type="episodic",
                importance=0.8,
                metadata={
                    "event_role": event["role"],
                    "turn_index": event["turn_index"],
                },
                namespace="tenant-a",
            ),
            namespace="tenant-a",
        )
        store.link_memory_to_session(
            session_id,
            memory_id,
            turn_index=int(event["turn_index"]),
            namespace="tenant-a",
        )

    first = consolidate_session(store, session_id=session_id, namespace="tenant-a")
    store.close()

    reopened = MemoryStore(db_path)
    reopened.initialize()
    try:
        second = consolidate_session(
            reopened,
            session_id=session_id,
            namespace="tenant-a",
        )

        assert first.to_dict() == second.to_dict()
        summaries = [
            memory
            for memory in reopened.get_session_memories(
                session_id,
                namespace="tenant-a",
            )
            if memory.metadata.get("episode_summary_for_session") == session_id
        ]
        assert len(summaries) == 1
        assert reopened.count_memories(namespace="tenant-a") == 4
    finally:
        reopened.close()
