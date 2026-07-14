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
        """Construct a relation without coercing unsafe public values."""
        if not isinstance(d, dict):
            raise TypeError("relation payload must be a mapping")

        def strict_int(name: str, value: Any, *, allow_none: bool = False) -> int | None:
            if value is None and allow_none:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"relation {name} must be an integer")
            return value

        relation_type = d.get("relation_type", "related_to")
        if not isinstance(relation_type, str):
            raise TypeError("relation type must be a string")
        confidence = d.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("relation confidence must be numeric")
        is_active = d.get("is_active", True)
        if not isinstance(is_active, bool):
            raise TypeError("relation is_active must be a boolean")
        return cls(
            id=strict_int("id", d.get("id"), allow_none=True),
            source_id=strict_int("source_id", d.get("source_id")),
            target_id=strict_int("target_id", d.get("target_id")),
            relation_type=relation_type,
            confidence=float(confidence),
            namespace=d.get("namespace"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            source=d.get("source"),
            is_active=is_active,
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
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievalEvidence:
        if not isinstance(d, dict):
            raise TypeError("evidence payload must be a mapping")
        matched_by = d.get("matched_by", ())
        if isinstance(matched_by, str) or not isinstance(matched_by, (list, tuple)):
            raise TypeError("evidence matched_by must be a sequence")
        return cls(
            score=d.get("score", 0.0),
            semantic_score=d.get("semantic_score", 0.0),
            recency_score=d.get("recency_score", 0.0),
            importance_score=d.get("importance_score", 0.0),
            strength_score=d.get("strength_score", 0.0),
            matched_by=tuple(matched_by),
            trust=d.get("trust", "unknown"),
            reason=d.get("reason", ""),
        )


@dataclass(frozen=True)
class EvolutionProposal:
    """Validated, serializable description of a proposed memory evolution."""

    candidate_id: int
    target_ids: tuple[int, ...]
    action: str
    relation_type: str
    metadata_patch: dict[str, Any]
    confidence: float
    actor: str
    reason: str
    namespace: str | None
    evidence: RetrievalEvidence

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "candidate_id": self.candidate_id,
            "target_ids": list(self.target_ids),
            "action": self.action,
            "relation_type": self.relation_type,
            "metadata_patch": self.metadata_patch,
            "confidence": self.confidence,
            "actor": self.actor,
            "reason": self.reason,
            "namespace": self.namespace,
            "evidence": self.evidence.to_dict(),
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionProposal:
        if not isinstance(d, dict):
            raise TypeError("evolution proposal payload must be a mapping")
        target_ids = d.get("target_ids", ())
        if isinstance(target_ids, str) or not isinstance(target_ids, (list, tuple)):
            raise TypeError("evolution target_ids must be a sequence")
        metadata_patch = d.get("metadata_patch", {})
        if not isinstance(metadata_patch, dict):
            raise TypeError("evolution metadata_patch must be a mapping")
        evidence = d.get("evidence", {})
        return cls(
            candidate_id=d.get("candidate_id"),
            target_ids=tuple(target_ids),
            action=d.get("action", ""),
            relation_type=d.get("relation_type", ""),
            metadata_patch=metadata_patch,
            confidence=d.get("confidence"),
            actor=d.get("actor", ""),
            reason=d.get("reason", ""),
            namespace=d.get("namespace"),
            evidence=(
                evidence
                if isinstance(evidence, RetrievalEvidence)
                else RetrievalEvidence.from_dict(evidence)
            ),
        )


@dataclass(frozen=True)
class EvolutionDecision:
    """Result of validating and applying (or rejecting) a proposal."""

    accepted: bool
    proposal: EvolutionProposal
    reason: str
    event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "accepted": self.accepted,
            "proposal": self.proposal.to_dict(),
            "reason": self.reason,
            "event_id": self.event_id,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionDecision:
        if not isinstance(d, dict):
            raise TypeError("evolution decision payload must be a mapping")
        proposal = d.get("proposal")
        if isinstance(proposal, EvolutionProposal):
            parsed = proposal
        elif isinstance(proposal, dict):
            parsed = EvolutionProposal.from_dict(proposal)
        else:
            raise TypeError("evolution decision proposal must be a mapping")
        event_id = d.get("event_id")
        if event_id is not None and (isinstance(event_id, bool) or not isinstance(event_id, int)):
            raise TypeError("evolution event_id must be an integer or null")
        return cls(
            accepted=bool(d.get("accepted", False)),
            proposal=parsed,
            reason=d.get("reason", ""),
            event_id=event_id,
        )


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
    relation_evidence: tuple[dict[str, Any], ...] = ()

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
            "relation_evidence": [dict(item) for item in self.relation_evidence],
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
