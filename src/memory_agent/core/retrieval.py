"""Memory retrieval engine with combined scoring and MMR diversity."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from memory_agent.core.config import RetrievalConfig
from memory_agent.core.embeddings import EmbeddingEngine
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord, RetrievalEvidence, SearchResult


class RetrievalEngine:
    """Retrieves memories using combined semantic, recency, importance,
    and strength scoring, with MMR diversity penalty.

    total_score = w_semantic * similarity + w_recency * recency_score
                + w_importance * importance + w_strength * strength
    """

    def __init__(
        self,
        store: MemoryStore,
        embeddings: EmbeddingEngine,
        config: RetrievalConfig | None = None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.config = config or RetrievalConfig()

    # ------------------------------------------------------------------
    # Individual scoring dimensions
    # ------------------------------------------------------------------

    def semantic_score(
        self,
        query_vec: np.ndarray,
        memory_id: int,
        *,
        namespace: str | None = None,
    ) -> float:
        """Cosine similarity between query and stored memory embedding."""
        blob = self.store.get_embedding(memory_id, namespace=namespace)
        if blob is None:
            return 0.0
        mem_vec = self.embeddings.decode_vector(blob[0])
        return self.embeddings.cosine_similarity(query_vec, mem_vec)

    def recency_score(self, hours_since_access: float) -> float:
        """Recency score: 1 / (1 + hours_since_access), normalized."""
        return 1.0 / (1.0 + hours_since_access)

    def strength_score(self, strength: float) -> float:
        """Direct use of recall strength."""
        return strength

    def importance_score(self, importance: float) -> float:
        """Direct use of importance."""
        return importance

    # ------------------------------------------------------------------
    # Combined scoring
    # ------------------------------------------------------------------

    def combined_score(
        self,
        memory: MemoryRecord,
        query_vec: np.ndarray | None = None,
        *,
        namespace: str | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        """Compute the combined retrieval score for a single memory."""
        sem = 0.0
        if query_vec is not None and memory.id is not None:
            sem = self.semantic_score(query_vec, memory.id, namespace=namespace)

        rec = self.recency_score(memory.hours_since_access)
        imp = self.importance_score(memory.importance)
        strg = self.strength_score(memory.strength)
        total = (
            self.config.w_semantic * sem
            + self.config.w_recency * rec
            + self.config.w_importance * imp
            + self.config.w_strength * strg
        )
        signal_threshold = (
            min_score if min_score is not None else self.config.min_score
        )
        matched_by = tuple(
            name
            for name, value in (
                ("semantic", sem),
                ("recency", rec),
                ("importance", imp),
                ("strength", strg),
            )
            if value >= signal_threshold
        )
        trust = self._trust_for(memory)
        if trust == "untrusted":
            confidence = memory.confidence
            reason = f"filtered: trust=untrusted (confidence={confidence:.3f})"
        else:
            signal_text = ", ".join(matched_by) or "none"
            reason = f"matched by {signal_text}; trust={trust}"
        evidence = RetrievalEvidence(
            score=total,
            semantic_score=sem,
            recency_score=rec,
            importance_score=imp,
            strength_score=strg,
            matched_by=matched_by,
            trust=trust,
            reason=reason,
        )
        return SearchResult(
            memory=memory,
            score=total,
            semantic_score=sem,
            recency_score=rec,
            importance_score=imp,
            strength_score=strg,
            evidence=evidence,
        )

    @staticmethod
    def _trust_for(memory: MemoryRecord) -> str:
        """Classify confidence without rejecting legacy unknown memories."""
        if memory.confidence is None:
            return "unknown"
        return "trusted" if memory.confidence >= 0.5 else "untrusted"

    # ------------------------------------------------------------------
    # Full retrieval
    # ------------------------------------------------------------------

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
    ) -> list[SearchResult]:
        """Retrieve the most relevant memories for a query.

        Candidate filtering, scoring, and MMR preserve the existing ranking
        contract while attaching deterministic evidence to each result.
        """
        top_k = top_k or self.config.top_k
        candidate_k = candidate_k or top_k
        candidate_k = max(candidate_k, top_k)
        min_score = min_score if min_score is not None else self.config.min_score
        mmr_lambda = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda

        # Encode query
        query_blob = self.embeddings.encode(query)
        query_vec = self.embeddings.decode_vector(query_blob)

        if memory_type:
            candidates = self.store.get_memories_by_type(
                memory_type, namespace=namespace
            )
        else:
            candidates = self.store.get_all_active_memories(namespace=namespace)

        if not candidates:
            return []

        results = [
            self.combined_score(
                mem, query_vec, namespace=namespace, min_score=min_score
            )
            for mem in candidates
        ]

        # Filter by minimum score
        results = [r for r in results if r.score >= min_score]

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        if use_mmr and len(results) > candidate_k:
            results = self._mmr_diversify(
                results, query_vec, mmr_lambda, candidate_k, namespace=namespace
            )
        else:
            results = results[:candidate_k]

        results = results[:top_k] if candidate_k == top_k else results

        # Batch-update access tracking for retrieved memories
        now_iso = datetime.now().isoformat()
        updates: list[tuple[Any, ...]] = []
        for r in results:
            if r.memory.id is not None:
                r.memory.access_count += 1
                r.memory.last_accessed_at = now_iso
                if namespace is None:
                    updates.append((now_iso, r.memory.access_count, r.memory.id))
                else:
                    updates.append(
                        (now_iso, r.memory.access_count, r.memory.id, namespace)
                    )
        if updates:
            if namespace is None:
                self.store.conn.executemany(
                    "UPDATE memories SET last_accessed_at = ?, access_count = ? WHERE id = ?",
                    updates,
                )
            else:
                self.store.conn.executemany(
                    "UPDATE memories SET last_accessed_at = ?, access_count = ? "
                    "WHERE id = ? AND namespace = ?",
                    updates,
                )
        self.store.conn.commit()

        return results

    # ------------------------------------------------------------------
    # Maximum Marginal Relevance (MMR)
    # ------------------------------------------------------------------

    def _mmr_diversify(
        self,
        candidates: list[SearchResult],
        query_vec: np.ndarray,
        mmr_lambda: float,
        top_k: int,
        *,
        namespace: str | None = None,
    ) -> list[SearchResult]:
        """Apply MMR to maximize diversity while maintaining relevance.

        MMR = argmax[ lambda * sim(q, m_i) - (1-lambda) * max sim(m_i, m_j) ]
        """
        if not candidates or top_k <= 0:
            return []

        selected: list[SearchResult] = []
        remaining = list(candidates)

        # Pick first: highest score
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = float("-inf")

            for i, cand in enumerate(remaining):
                # Relevance component
                relevance = cand.score

                # Diversity component: max similarity to any selected
                max_sim_to_selected = 0.0
                if selected and cand.memory.id is not None:
                    cand_vec = self._get_vec(cand.memory.id, namespace=namespace)
                    if cand_vec is not None:
                        for sel in selected:
                            if sel.memory.id is not None:
                                sel_vec = self._get_vec(
                                    sel.memory.id, namespace=namespace
                                )
                                if sel_vec is not None:
                                    sim = self.embeddings.cosine_similarity(
                                        cand_vec, sel_vec
                                    )
                                    max_sim_to_selected = max(
                                        max_sim_to_selected, sim
                                    )
                mmr = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _get_vec(
        self, memory_id: int, *, namespace: str | None = None
    ) -> np.ndarray | None:
        """Get decoded vector for a memory (with caching)."""
        blob = self.store.get_embedding(memory_id, namespace=namespace)
        if blob is None:
            return None
        return self.embeddings.decode_vector(blob[0])
