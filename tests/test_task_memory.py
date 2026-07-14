"""RED contracts for procedural task-memory packs and bounded task context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord
from memory_agent.core.task_memory import (
    TaskMemoryPack,
    TaskMemoryPackStore,
    build_task_context,
)


@pytest.fixture
def memory_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "task-memory.db")
    store.initialize()
    try:
        yield store
    finally:
        store.close()


def _pack(
    task_name: str = "deploy-service",
    *,
    namespace: str = "tenant-a",
    instructions: list[str] | None = None,
    required_memory_ids: list[int] | None = None,
) -> TaskMemoryPack:
    return TaskMemoryPack(
        task_name=task_name,
        triggers=["deploy", "release"],
        instructions=instructions or ["run checks", "deploy the approved artifact"],
        constraints=["never deploy an unapproved artifact"],
        required_memory_ids=required_memory_ids or [],
        successful_examples=["release 42 completed with a clean rollback plan"],
        confidence=0.92,
        namespace=namespace,
    )


def test_task_memory_pack_round_trips_as_strict_json_safe_payload() -> None:
    pack = _pack()

    payload = pack.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)

    assert json.loads(encoded) == payload
    assert set(payload) >= {
        "task_name",
        "triggers",
        "instructions",
        "constraints",
        "required_memory_ids",
        "successful_examples",
        "confidence",
        "namespace",
    }
    assert TaskMemoryPack.from_dict(payload) == pack


def test_task_pack_store_isolates_namespaces_and_persists_lifecycle(
    memory_store: MemoryStore,
) -> None:
    packs = TaskMemoryPackStore(memory_store)
    tenant_a = _pack(namespace="tenant-a")
    tenant_b = _pack(namespace="tenant-b")

    first_id = packs.save(tenant_a)
    packs.save(tenant_b)

    assert packs.get("deploy-service", namespace="tenant-a") == tenant_a
    assert packs.get("deploy-service", namespace="tenant-b") == tenant_b
    assert packs.get("deploy-service", namespace="tenant-c") is None

    # Lifecycle is stored in the same SQLite-backed records and survives a
    # fresh store facade. Archived and superseded packs are not candidates for
    # active task retrieval.
    memory_store.archive_memory(
        first_id,
        reason="stale procedure",
        namespace="tenant-a",
    )
    assert packs.list("deploy-service", namespace="tenant-a") == []

    replacement = _pack(namespace="tenant-a", instructions=["use the new release workflow"])
    replacement_id = packs.save(replacement)
    old = memory_store.get_memory(first_id, namespace="tenant-a")
    assert old is not None
    old.superseded_by = replacement_id
    memory_store.update_memory(old, namespace="tenant-a")
    superseded_id = packs.save(
        _pack(namespace="tenant-a", instructions=["use the superseded workflow"])
    )
    superseded = memory_store.get_memory(superseded_id, namespace="tenant-a")
    assert superseded is not None
    superseded.superseded_by = replacement_id
    memory_store.update_memory(superseded, namespace="tenant-a")


    # A superseded procedural record is excluded even while still active.
    assert [pack.instructions for pack in packs.list("deploy-service", namespace="tenant-a")] == [
        replacement.instructions
    ]


def test_build_task_context_expands_required_memories_with_bounded_budget(
    memory_store: MemoryStore,
) -> None:
    required_id = memory_store.add_memory(
        MemoryRecord(
            content="The production deployment window is 15 minutes.",
            memory_type="semantic",
            importance=0.9,
            namespace="tenant-a",
        ),
        namespace="tenant-a",
    )
    irrelevant_id = memory_store.add_memory(
        MemoryRecord(
            content="An unrelated note that must not be pulled into this task context.",
            memory_type="semantic",
            importance=1.0,
            namespace="tenant-a",
        ),
        namespace="tenant-a",
    )
    packs = TaskMemoryPackStore(memory_store)
    packs.save(_pack(required_memory_ids=[required_id]))

    packet = build_task_context(
        "deploy-service",
        "release the approved artifact",
        namespace="tenant-a",
        store=packs,
        budget_chars=80,
        reserved_chars=10,
    )

    selected_ids = packet.selected_ids
    assert required_id in selected_ids
    assert irrelevant_id not in selected_ids
    assert packet.used_chars <= packet.available_chars
    assert packet.limit == 80
    assert packet.omitted or packet.used_chars < packet.available_chars
