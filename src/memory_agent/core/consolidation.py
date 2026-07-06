"""Deterministic memory consolidation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from memory_agent.core.config import ConsolidationConfig
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


class TextSimilarity(Protocol):
    def similarity(self, left: str, right: str) -> float:
        """Return similarity in the inclusive range [0.0, 1.0]."""


class ConsolidationAction(str, Enum):
    CREATE = "create"
    REINFORCE = "reinforce"
    UPDATE = "update"
    IGNORE = "ignore"


@dataclass
class ConsolidationDecision:
    action: ConsolidationAction
    candidate: MemoryRecord
    existing_memory_id: int | None = None
    new_memory_id: int | None = None
    reason: str = ""


class MemoryConsolidator:
    """Consolidates extracted memories before they are stored."""

    def __init__(
        self,
        store: MemoryStore,
        similarity: TextSimilarity,
        config: ConsolidationConfig,
    ) -> None:
        self.store = store
        self.similarity = similarity
        self.config = config

    def consolidate(self, candidate: MemoryRecord) -> ConsolidationDecision:
        if candidate.importance < self.config.auto_consolidate_threshold:
            return ConsolidationDecision(
                action=ConsolidationAction.IGNORE,
                candidate=candidate,
                reason="candidate below auto-consolidation threshold",
            )

        active = self.store.get_all_active_memories()
        same_type = [m for m in active if m.memory_type == candidate.memory_type]
        best = self._best_match(candidate, same_type)

        if best is None:
            new_id = self.store.add_memory(candidate, commit=False)
            return ConsolidationDecision(
                action=ConsolidationAction.CREATE,
                candidate=candidate,
                new_memory_id=new_id,
                reason="no active memory matched candidate",
            )

        existing, score = best
        if self._supersedes(existing, candidate, score):
            new_id = self.store.add_memory(candidate, commit=False)
            candidate.metadata = {
                **candidate.metadata,
                "supersedes": existing.id,
                "consolidation_action": "update",
            }
            self.store.update_memory(candidate, commit=False)
            self.store.archive_memory(
                existing.id or -1,
                reason="superseded",
                metadata={
                    "superseded_by": new_id,
                    "superseded_at": datetime.now().isoformat(),
                },
                commit=False,
            )
            return ConsolidationDecision(
                action=ConsolidationAction.UPDATE,
                candidate=candidate,
                existing_memory_id=existing.id,
                new_memory_id=new_id,
                reason=f"candidate superseded active memory at similarity {score:.3f}",
            )

        if score >= self.config.duplicate_similarity_threshold:
            existing.strength = min(1.0, existing.strength + 0.15)
            existing.access_count += 1
            existing.last_accessed_at = datetime.now().isoformat()
            existing.metadata = {
                **existing.metadata,
                "consolidation_action": "reinforce",
                "last_duplicate": candidate.content,
            }
            self.store.update_memory(existing, commit=False)
            return ConsolidationDecision(
                action=ConsolidationAction.REINFORCE,
                candidate=candidate,
                existing_memory_id=existing.id,
                reason=f"duplicate similarity {score:.3f}",
            )

        new_id = self.store.add_memory(candidate, commit=False)
        return ConsolidationDecision(
            action=ConsolidationAction.CREATE,
            candidate=candidate,
            new_memory_id=new_id,
            reason=f"best similarity {score:.3f} did not trigger consolidation",
        )

    def forget_matching(self, query: str) -> int:
        archived = 0
        for memory in self.store.get_all_active_memories():
            score = self.similarity.similarity(query, memory.content)
            if score >= self.config.explicit_forget_min_score and memory.id is not None:
                self.store.archive_memory(
                    memory.id,
                    reason="explicit_user_request",
                    metadata={"forget_query": query},
                    commit=False,
                )
                archived += 1
        return archived

    def _best_match(
        self,
        candidate: MemoryRecord,
        memories: list[MemoryRecord],
    ) -> tuple[MemoryRecord, float] | None:
        best_memory: MemoryRecord | None = None
        best_score = 0.0
        for memory in memories:
            score = self.similarity.similarity(memory.content, candidate.content)
            if score > best_score:
                best_memory = memory
                best_score = score
        if best_memory is None:
            return None
        return best_memory, best_score

    def _supersedes(self, existing: MemoryRecord, candidate: MemoryRecord, score: float) -> bool:
        if existing.memory_type != "preference" or candidate.memory_type != "preference":
            return False
        if score < self.config.supersede_similarity_threshold:
            return False
        existing_polarity = existing.metadata.get("polarity")
        candidate_polarity = candidate.metadata.get("polarity")
        if existing_polarity and candidate_polarity:
            return existing_polarity != candidate_polarity
        return False
