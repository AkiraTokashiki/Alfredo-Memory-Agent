"""Memory models and data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "strength": self.strength,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": json.dumps(self.metadata),
            "tags": self.tags,
            "is_active": 1 if self.is_active else 0,
        }

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

    @property
    def estimated_chars(self) -> int:
        """Approximate formatted context cost for this memory."""
        return len(self.memory.content)


@dataclass
class SessionRecord:
    """A conversation session."""

    id: int | None = None
    label: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    memory_ids: list[int] = field(default_factory=list)


@dataclass
class AgentState:
    """The full state of the memory agent."""

    session_id: int | None = None
    current_context: list[MemoryRecord] = field(default_factory=list)
    turn_count: int = 0
    session_memories: int = 0
    total_memories: int = 0
