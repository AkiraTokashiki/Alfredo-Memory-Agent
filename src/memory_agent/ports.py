"""Public dependency-injection contracts for memory components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import (
    EvolutionDecision,
    EvolutionProposal,
    MemoryRecord,
    MemoryRelation,
    RetrievalEvidence,
    SearchResult,
)
 
 
@runtime_checkable
class EvolutionPlannerPort(Protocol):
    """Deterministic proposal generator for memory evolution."""

    def propose(
        self,
        candidate: MemoryRecord,
        neighbors: list[MemoryRecord],
        context: dict[str, Any],
    ) -> EvolutionProposal | None: ...


@runtime_checkable
class MemoryStorePort(Protocol):
    """Persistence operations required by the memory lifecycle."""


    def initialize(self) -> None: ...
    def close(self) -> None: ...

    def commit(self) -> None: ...
    def add_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int: ...

    def get_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> MemoryRecord | None: ...

    def update_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def delete_memory(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        namespace: str | None = None,
    ) -> None: ...

    def add_relation(
        self,
        relation: MemoryRelation,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int: ...

    def get_relations(
        self,
        source_id: int | None = None,
        *,
        namespace: str | None = None,
        active_only: bool = True,
        target_id: int | None = None,
        relation_type: str | None = None,
    ) -> list[MemoryRelation]: ...

    def deactivate_relation(
        self,
        relation_id: int,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def get_all_active_memories(
        self, *, namespace: str | None = None
    ) -> list[MemoryRecord]: ...
    def get_memories_by_type(
        self, memory_type: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]: ...

    def get_embedding(
        self, memory_id: int, *, namespace: str | None = None
    ) -> tuple[bytes, str] | None: ...

    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def count_memories(
        self, *, active_only: bool = True, namespace: str | None = None
    ) -> int: ...

    def create_session(
        self,
        label: str = "",
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int: ...

    def end_session(
        self,
        session_id: int,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def link_memory_to_session(
        self,
        session_id: int,
        memory_id: int,
        turn_index: int | None = None,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def save_embedding(
        self,
        memory_id: int,
        embedding: bytes,
        model_name: str,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def update_strengths(
        self,
        updates: list[tuple[float, int]],
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None: ...

    def archive_below_threshold(
        self,
        threshold: float,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int: ...
    def get_embedding_count(self, *, namespace: str | None = None) -> int: ...

    def record_access(
        self,
        accesses: list[tuple[int, int]],
        *,
        namespace: str | None = None,
        accessed_at: str | None = None,
        commit: bool = True,
    ) -> None: ...
    def apply_evolution(self, proposal: EvolutionProposal) -> EvolutionDecision: ...

@runtime_checkable
class EmbeddingPort(Protocol):
    """Text embedding provider used to index and compare memories."""

    model_name: str

    def encode(self, text: str) -> bytes:
        """Encode text into a provider-specific serialized vector."""
        ...
    def decode_vector(self, blob: bytes) -> Any:
        """Decode a provider-specific vector representation."""
        ...

    def cosine_similarity(self, left: Any, right: Any) -> float:
        """Return cosine similarity for two decoded vectors."""
        ...


@runtime_checkable
class RetrievalPort(Protocol):
    """Candidate retrieval operation exposed to the orchestrator."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        memory_type: str | None = None,
        min_score: float | None = None,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
        candidate_k: int | None = None,
        namespace: str | None = None,
        commit: bool = True,
        record_access: bool = True,
        include_related: bool = False,
    ) -> list[SearchResult]:
        """Return ranked candidates and optional relation neighbors."""
        ...


@runtime_checkable
class TrustPolicyPort(Protocol):
    """Trust decision operation applied before context injection."""

    def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
        """Explain whether a memory is trusted for retrieval."""
        ...
