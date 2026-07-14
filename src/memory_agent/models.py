"""Memory models and data structures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Copy supported values into JSON-safe Python primitives."""
    seen = _seen if _seen is not None else set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON-safe floats must be finite")
        return value
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("JSON-safe values cannot contain cycles")
        seen.add(marker)
        try:
            return [_json_safe(item, seen) for item in value]
        finally:
            seen.remove(marker)
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            raise ValueError("JSON-safe values cannot contain cycles")
        seen.add(marker)
        try:
            if not all(isinstance(key, str) for key in value):
                raise TypeError("JSON-safe object keys must be strings")
            return {key: _json_safe(item, seen) for key, item in value.items()}
        finally:
            seen.remove(marker)
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


@dataclass
class MemoryRecord:
    """A single memory entry."""

    id: int | None = None
    content: str = ""
    memory_type: str = "episodic"  # episodic | semantic | procedural | preference
    importance: float = 0.5
    strength: float = 1.0
    access_count: int = 0
    last_accessed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_active: bool = True
    namespace: str | None = None
    confidence: float | None = None
    sensitivity: str | None = None
    source: str | None = None
    superseded_by: int | str | None = None
    last_decision_reason: str | None = None

    @property
    def age_hours(self) -> float:
        """Hours since this memory was created."""
        if not self.created_at:
            return 0.0
        created = datetime.fromisoformat(self.created_at)
        return (datetime.now() - created).total_seconds() / 3600

    @property
    def hours_since_access(self) -> float:
        """Hours since this memory was last accessed."""
        if not self.last_accessed_at:
            return self.age_hours
        accessed = datetime.fromisoformat(self.last_accessed_at)
        return (datetime.now() - accessed).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe public representation."""
        return _json_safe({
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "strength": self.strength,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "is_active": self.is_active,
            "namespace": self.namespace,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity,
            "source": self.source,
            "superseded_by": self.superseded_by,
            "last_decision_reason": self.last_decision_reason,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryRecord:
        metadata = d.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        tags = d.get("tags", [])
        if isinstance(tags, str):
            tags = json.loads(tags) if tags else []
        return cls(
            id=d.get("id"),
            content=d.get("content", ""),
            memory_type=d.get("memory_type", "episodic"),
            importance=float(d.get("importance", 0.5)),
            strength=float(d.get("strength", 1.0)),
            access_count=int(d.get("access_count", 0)),
            last_accessed_at=d.get("last_accessed_at"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            metadata=metadata,
            tags=tags,
            is_active=bool(d.get("is_active", 1)),
            namespace=d.get("namespace"),
            confidence=(
                float(d["confidence"])
                if d.get("confidence") is not None
                else None
            ),
            sensitivity=d.get("sensitivity"),
            source=d.get("source"),
            superseded_by=d.get("superseded_by"),
            last_decision_reason=d.get("last_decision_reason"),
        )


@dataclass
class MemoryRelation:
    """A typed, namespace-scoped edge between two memories."""

    id: int | None = None
    source_id: int = 0
    target_id: int = 0
    relation_type: str = "related_to"
    confidence: float = 1.0
    namespace: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    source: str | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe public representation."""
        return _json_safe({
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "is_active": self.is_active,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryRelation:
        """Construct a relation from a JSON/SQLite-style mapping."""
        confidence = d.get("confidence", 1.0)
        return cls(
            id=int(d["id"]) if d.get("id") is not None else None,
            source_id=int(d["source_id"]),
            target_id=int(d["target_id"]),
            relation_type=str(d.get("relation_type", "related_to")),
            confidence=float(confidence),
            namespace=d.get("namespace"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            source=d.get("source"),
            is_active=bool(d.get("is_active", 1)),
        )


@dataclass(frozen=True)
class RetrievalEvidence:
    """Explain why a memory was selected or rejected during retrieval."""

    score: float = 0.0
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    strength_score: float = 0.0
    matched_by: tuple[str, ...] = ()
    trust: str = "unknown"
    reason: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "score",
            "semantic_score",
            "recency_score",
            "importance_score",
            "strength_score",
        ):
            value = getattr(self, field_name)
            try:
                is_finite = math.isfinite(value)
            except TypeError as exc:
                raise ValueError(f"{field_name} must be finite") from exc
            if not is_finite:
                raise ValueError(f"{field_name} must be finite")
        object.__setattr__(self, "matched_by", tuple(self.matched_by))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe public representation."""
        return _json_safe({
            "score": self.score,
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "strength_score": self.strength_score,
            "matched_by": list(self.matched_by),
            "trust": self.trust,
            "reason": self.reason,
        })


@dataclass
class SearchResult:
    """A memory returned from a search query."""

    memory: MemoryRecord
    score: float
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    strength_score: float = 0.0
    evidence: RetrievalEvidence | None = None

    @property
    def estimated_chars(self) -> int:
        """Approximate formatted context cost for this memory."""
        return len(self.memory.content)
    @property
    def matched_by(self) -> tuple[str, ...]:
        """Signals that contributed to this result."""
        return self.evidence.matched_by if self.evidence else ()

    @property
    def trust(self) -> str:
        """Trust classification attached to this result."""
        if self.evidence is not None:
            return self.evidence.trust
        if self.memory.confidence is None:
            return "unknown"
        return "trusted" if self.memory.confidence >= 0.5 else "untrusted"

    @property
    def reason(self) -> str:
        """Deterministic explanation for this result."""
        return self.evidence.reason if self.evidence else ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe public representation."""
        return _json_safe({
            "memory": self.memory.to_dict(),
            "score": self.score,
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "strength_score": self.strength_score,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        })


@dataclass
class SessionRecord:
    """A conversation session."""

    id: int | None = None
    label: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    memory_ids: list[int] = field(default_factory=list)
    namespace: str | None = None



@dataclass
class AgentState:
    """The full state of the memory agent."""

    session_id: int | None = None
    current_context: list[MemoryRecord] = field(default_factory=list)
    turn_count: int = 0
    session_memories: int = 0
    total_memories: int = 0
