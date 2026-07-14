"""Deterministic episodic summaries and idempotent session consolidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from memory_agent.models import MemoryRecord


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    raise TypeError(f"unsupported summary value: {type(value).__name__}")


@dataclass(frozen=True)
class EpisodeSummary:
    """Stable, serializable result of consolidating one session."""

    session_id: int
    namespace: str | None
    summary: str
    event_count: int = 0
    event_ids: tuple[int, ...] = ()
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _safe(
            {
                "session_id": self.session_id,
                "namespace": self.namespace,
                "summary": self.summary,
                "event_count": self.event_count,
                "event_ids": list(self.event_ids),
                "idempotency_key": self.idempotency_key,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeSummary":
        if not isinstance(payload, dict):
            raise TypeError("episode summary payload must be a mapping")
        return cls(
            session_id=int(payload.get("session_id", 0)),
            namespace=payload.get("namespace"),
            summary=str(payload.get("summary", "")),
            event_count=int(payload.get("event_count", 0)),
            event_ids=tuple(int(item) for item in payload.get("event_ids", [])),
            idempotency_key=str(payload.get("idempotency_key", "")),
        )


class EpisodeSummaryBuilder:
    """Build summaries without clocks, randomness, or remote model calls."""

    def build(
        self,
        events: Iterable[dict[str, Any]],
        *,
        session_id: int,
        namespace: str | None = None,
    ) -> EpisodeSummary:
        normalized = []
        for ordinal, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise TypeError("episode events must be mappings")
            content = str(event.get("content", "")).strip()
            if not content:
                continue
            turn = event.get("turn_index", ordinal)
            try:
                turn = int(turn)
            except (TypeError, ValueError):
                turn = ordinal
            role = str(event.get("role", event.get("event_role", "unknown"))).strip() or "unknown"
            event_id = event.get("id")
            normalized.append((turn, ordinal, role, content, event_id))
        normalized.sort(key=lambda item: (item[0], item[1]))
        lines = [f"Turn {turn} ({role}): {content}" for turn, _, role, content, _ in normalized]
        summary = " ".join(lines)
        ids = tuple(int(item[4]) for item in normalized if isinstance(item[4], int) and not isinstance(item[4], bool))
        key_material = {
            "session_id": session_id,
            "namespace": namespace,
            "events": [
                {"turn_index": turn, "role": role, "content": content, "id": event_id}
                for turn, _, role, content, event_id in normalized
            ],
        }
        key = hashlib.sha256(
            json.dumps(key_material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        return EpisodeSummary(
            session_id=session_id,
            namespace=namespace,
            summary=summary,
            event_count=len(normalized),
            event_ids=ids,
            idempotency_key=key,
        )


def _event_dict(memory: MemoryRecord) -> dict[str, Any]:
    metadata = memory.metadata or {}
    return {
        "id": memory.id,
        "turn_index": metadata.get("turn_index", 0),
        "role": metadata.get("event_role", metadata.get("role", "unknown")),
        "content": memory.content,
    }


def consolidate_session(
    store: Any,
    *,
    session_id: int,
    namespace: str | None = None,
    builder: EpisodeSummaryBuilder | None = None,
) -> EpisodeSummary:
    """Consolidate session events into exactly one persisted summary memory."""
    if isinstance(session_id, bool) or not isinstance(session_id, int):
        raise TypeError("session_id must be an integer")
    builder = builder or EpisodeSummaryBuilder()
    memories = store.get_session_memories(session_id, namespace=namespace)
    event_memories = [
        memory
        for memory in memories
        if memory.metadata.get("episode_summary_for_session") != session_id
    ]
    events = [_event_dict(memory) for memory in event_memories]
    summary = builder.build(events, session_id=session_id, namespace=namespace)

    existing = [
        memory
        for memory in memories
        if memory.metadata.get("episode_summary_for_session") == session_id
    ]
    if existing:
        record = existing[0]
        # Update in place if new events were linked after an earlier close. This
        # preserves one summary row and keeps repeated calls idempotent.
        if record.metadata.get("episode_idempotency_key") != summary.idempotency_key or record.content != summary.summary:
            record.content = summary.summary
            record.metadata = {
                **record.metadata,
                "episode_idempotency_key": summary.idempotency_key,
                "episode_event_ids": list(summary.event_ids),
                "event_count": summary.event_count,
            }
            store.update_memory(record, namespace=namespace)
        return summary

    record = MemoryRecord(
        content=summary.summary,
        memory_type="episodic",
        importance=0.7,
        confidence=1.0,
        namespace=namespace,
        metadata={
            "episode_summary_for_session": session_id,
            "episode_idempotency_key": summary.idempotency_key,
            "episode_event_ids": list(summary.event_ids),
            "event_count": summary.event_count,
            "lifecycle": "active",
        },
        tags=["episode-summary"],
    )
    memory_id = store.add_memory(record, namespace=namespace, commit=False)
    store.link_memory_to_session(
        session_id,
        memory_id,
        turn_index=(max((int(event.get("turn_index", 0)) for event in events), default=0) + 1),
        namespace=namespace,
        commit=False,
    )
    store.commit()
    return summary
