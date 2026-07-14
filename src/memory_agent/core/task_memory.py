"""Procedural task-memory packs and bounded task context construction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from memory_agent.core.context_budget import ContextBudgetPacker, RecallPacket
from memory_agent.models import MemoryRecord, SearchResult


def _json_safe(value: Any) -> Any:
    """Return a strict JSON-safe value, rejecting non-finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON-safe floats must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON-safe object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


@dataclass
class TaskMemoryPack:
    """A reusable, namespace-scoped procedural instruction pack."""

    task_name: str
    triggers: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    required_memory_ids: list[int] = field(default_factory=list)
    successful_examples: list[str] = field(default_factory=list)
    confidence: float = 0.5
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str):
            raise TypeError("task_name must be a string")
        if not math.isfinite(float(self.confidence)):
            raise ValueError("confidence must be finite")
        self.confidence = float(self.confidence)
        for name in (
            "triggers",
            "instructions",
            "constraints",
            "successful_examples",
        ):
            values = getattr(self, name)
            if isinstance(values, str):
                raise TypeError(f"{name} must be a sequence of strings")
            if not all(isinstance(value, str) for value in values):
                raise TypeError(f"{name} must contain only strings")
            setattr(self, name, list(values))
        if isinstance(self.required_memory_ids, (str, bytes)):
            raise TypeError("required_memory_ids must be a sequence of integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.required_memory_ids
        ):
            raise TypeError("required_memory_ids must contain only integers")
        self.required_memory_ids = list(dict.fromkeys(self.required_memory_ids))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "task_name": self.task_name,
                "triggers": list(self.triggers),
                "instructions": list(self.instructions),
                "constraints": list(self.constraints),
                "required_memory_ids": list(self.required_memory_ids),
                "successful_examples": list(self.successful_examples),
                "confidence": self.confidence,
                "namespace": self.namespace,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskMemoryPack":
        if not isinstance(payload, dict):
            raise TypeError("task memory pack payload must be a mapping")
        return cls(
            task_name=payload.get("task_name", ""),
            triggers=list(payload.get("triggers", [])),
            instructions=list(payload.get("instructions", [])),
            constraints=list(payload.get("constraints", [])),
            required_memory_ids=list(payload.get("required_memory_ids", [])),
            successful_examples=list(payload.get("successful_examples", [])),
            confidence=payload.get("confidence", 0.5),
            namespace=payload.get("namespace"),
        )


class TaskMemoryPackStore:
    """SQLite-backed persistence facade for procedural task packs."""

    def __init__(self, store: Any) -> None:
        self.memory_store = store
        # ``store`` is a convenient public alias used by context builders.
        self.store = store

    def save(self, pack: TaskMemoryPack) -> int:
        if not isinstance(pack, TaskMemoryPack):
            raise TypeError("pack must be a TaskMemoryPack")
        payload = pack.to_dict()
        memory = MemoryRecord(
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            memory_type="procedural",
            importance=max(0.0, min(1.0, pack.confidence)),
            confidence=pack.confidence,
            namespace=pack.namespace,
            metadata={
                "task_name": pack.task_name,
                "triggers": list(pack.triggers),
                "constraints": list(pack.constraints),
                "required_memory_ids": list(pack.required_memory_ids),
                "successful_examples": list(pack.successful_examples),
                "lifecycle": "active",
            },
            tags=["task-memory", pack.task_name],
        )
        return self.memory_store.add_memory(memory, namespace=pack.namespace)

    def _is_candidate(self, memory: MemoryRecord, task_name: str) -> bool:
        if memory.memory_type != "procedural" or not memory.is_active:
            return False
        if memory.superseded_by is not None:
            return False
        metadata = memory.metadata or {}
        if metadata.get("task_name") != task_name:
            return False
        if metadata.get("stale") is True:
            return False
        if str(metadata.get("lifecycle", "active")).lower() not in {"active", "current"}:
            return False
        if str(metadata.get("status", "active")).lower() in {
            "stale",
            "archived",
            "superseded",
            "expired",
            "forgotten",
        }:
            return False
        return True

    def _records(self, task_name: str, namespace: str | None) -> list[MemoryRecord]:
        records = self.memory_store.get_memories_by_type("procedural", namespace=namespace)
        return [record for record in records if self._is_candidate(record, task_name)]

    @staticmethod
    def _unpack(record: MemoryRecord) -> TaskMemoryPack:
        try:
            payload = json.loads(record.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = dict(record.metadata)
            payload.setdefault("task_name", record.metadata.get("task_name", ""))
            payload.setdefault("namespace", record.namespace)
            payload.setdefault("confidence", record.confidence if record.confidence is not None else 0.5)
        payload["namespace"] = record.namespace
        if record.confidence is not None:
            payload["confidence"] = record.confidence
        return TaskMemoryPack.from_dict(payload)

    def get(self, task_name: str, *, namespace: str | None = None) -> TaskMemoryPack | None:
        records = self._records(task_name, namespace)
        return self._unpack(records[0]) if records else None

    def list(self, task_name: str, *, namespace: str | None = None) -> list[TaskMemoryPack]:
        return [self._unpack(record) for record in self._records(task_name, namespace)]

    def get_record(self, task_name: str, *, namespace: str | None = None) -> MemoryRecord | None:
        records = self._records(task_name, namespace)
        return records[0] if records else None


def _pack_context_text(pack: TaskMemoryPack) -> str:
    parts = [f"Task: {pack.task_name}"]
    if pack.instructions:
        parts.append("Instructions: " + "; ".join(pack.instructions))
    if pack.constraints:
        parts.append("Constraints: " + "; ".join(pack.constraints))
    if pack.successful_examples:
        parts.append("Examples: " + "; ".join(pack.successful_examples))
    return "\n".join(parts)


def build_task_context(
    task: str,
    query: str,
    *,
    namespace: str | None,
    store: TaskMemoryPackStore,
    budget_chars: int,
    reserved_chars: int = 0,
) -> RecallPacket:
    """Build a bounded packet from one active pack and its required memories.

    No global keyword retrieval is performed: only IDs explicitly named by the
    pack and active, same-namespace relation neighbors are considered.
    """
    del query  # selection is intentionally explicit and deterministic
    pack = store.get(task, namespace=namespace)
    packer = ContextBudgetPacker(budget_chars=budget_chars, reserved_chars=reserved_chars)
    if pack is None:
        return packer.pack([])

    memory_store = store.memory_store
    required: list[MemoryRecord] = []
    seen: set[int] = set()
    for memory_id in pack.required_memory_ids:
        memory = memory_store.get_memory(memory_id, namespace=namespace)
        if memory is None or not memory.is_active or memory.superseded_by is not None:
            continue
        if memory.id is not None and memory.id not in seen:
            required.append(memory)
            seen.add(memory.id)
        # Expand only edges touching explicitly required memories. Both
        # directions are valid because relation edges are directional.
        relations = list(
            memory_store.get_relations(
                memory_id, namespace=namespace, active_only=True
            )
        )
        relations.extend(
            relation
            for relation in memory_store.get_relations(
                target_id=memory_id, namespace=namespace, active_only=True
            )
            if relation.source_id != memory_id
        )
        for relation in relations:
            neighbor_id = (
                relation.target_id
                if relation.source_id == memory_id
                else relation.source_id
            )
            if neighbor_id in seen:
                continue
            neighbor = memory_store.get_memory(neighbor_id, namespace=namespace)
            if (
                neighbor is None
                or not neighbor.is_active
                or neighbor.superseded_by is not None
                or neighbor.namespace != namespace
            ):
                continue
            required.append(neighbor)
            seen.add(neighbor_id)

    results: list[SearchResult] = []
    # The pack itself explains the procedure but is not assigned a memory ID;
    # selected_ids therefore remains a precise list of supporting memories.
    pack_record = MemoryRecord(
        content=_pack_context_text(pack),
        memory_type="procedural",
        importance=pack.confidence,
        confidence=pack.confidence,
        namespace=namespace,
    )
    results.append(SearchResult(memory=pack_record, score=1.0))
    for index, memory in enumerate(required):
        # Required memories outrank relation expansion while remaining bounded.
        score = 0.9 if index < len(pack.required_memory_ids) else 0.5
        results.append(SearchResult(memory=memory, score=score))
    packet = packer.pack(results)
    # Remove the synthetic pack from selected IDs without affecting budget data.
    return packet
