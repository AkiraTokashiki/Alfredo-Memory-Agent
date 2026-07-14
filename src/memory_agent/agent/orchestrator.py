"""MemoryAgent orchestrator — the main agent loop.

Cycle:
1. Perceive — receive user input
2. Extract — analyze input for extractable memories
3. Retrieve — search memories for relevant context
4. Format context — build augmented prompt with recollections
5. Memorize — store extracted memories and interaction
6. Decay — apply forgetting curve
7. Return augmented context + agent response
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_agent.agent.decision import (
    extract_forget_query,
    extract_from_input,
    should_remember,
    summarize_interaction,
)
from memory_agent.core.evolution import OfflineEvolutionPlanner
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.embeddings import create_embedding_engine
from memory_agent.core.consolidation import (
    ConsolidationAction,
    ConsolidationDecision,
    MemoryConsolidator,
)
from memory_agent.core.context_budget import ContextBudgetPacker, RecallPacket
from memory_agent.core.forgetting import ForgettingCurve
from memory_agent.core.memory_store import MemoryStore
from memory_agent.core.retrieval import RetrievalEngine
from memory_agent.models import (
    AgentState,
    EvolutionDecision,
    MemoryRecord,
    RetrievalEvidence,
    SearchResult,
)
from memory_agent.ports import (
    EmbeddingPort,
    EvolutionPlannerPort,
    MemoryStorePort,
    RetrievalPort,
    TrustPolicyPort,
)


class DefaultTrustPolicy:
    """Conservative default policy for legacy memories without confidence."""

    def __init__(self, minimum_confidence: float = 0.5) -> None:
        self.minimum_confidence = minimum_confidence

    def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
        if memory.confidence is None:
            trust = "unknown"
        elif memory.confidence >= self.minimum_confidence:
            trust = "trusted"
        else:
            trust = "untrusted"
        return RetrievalEvidence(
            trust=trust,
            reason=f"default confidence policy: {trust}",
        )

class EmbeddingSimilarity:
    """Text similarity adapter backed by the configured embedding engine."""

    def __init__(self, embeddings: EmbeddingEngine) -> None:
        self.embeddings = embeddings

    def similarity(self, left: str, right: str) -> float:
        left_vec = self.embeddings.decode_vector(self.embeddings.encode(left))
        right_vec = self.embeddings.decode_vector(self.embeddings.encode(right))
        semantic = self.embeddings.cosine_similarity(left_vec, right_vec)
        return max(semantic, self._topic_similarity(left, right))

    def _topic_similarity(self, left: str, right: str) -> float:
        left_terms = set(self._normalize(left).split())
        right_terms = set(self._normalize(right).split())
        if not left_terms or not right_terms:
            return 0.0
        overlap = left_terms & right_terms
        if not overlap:
            return 0.0
        return len(overlap) / min(len(left_terms), len(right_terms))

    def _normalize(self, text: str) -> str:
        lowered = text.lower()
        for prefix in (
            "the user prefers:",
            "the user does not like:",
            "the user usually:",
            "fact:",
            "user:",
            "response:",
        ):
            lowered = lowered.replace(prefix, " ")
        return " ".join(lowered.split())


class MemoryAgent:
    """Main orchestrator for the memory agent system.

    Manages the full perceiving → remembering → deciding lifecycle.
    """

    def __init__(
        self,
        config: MemoryAgentConfig | None = None,
        db_path: str | Path | None = None,
        *,
        store: MemoryStorePort | None = None,
        embedder: EmbeddingPort | None = None,
        retrieval: RetrievalPort | None = None,
        trust_policy: TrustPolicyPort | None = None,
        evolution_planner: EvolutionPlannerPort | None = None,
    ):
        self.config = config or MemoryAgentConfig.default()

        # Resolve db path for legacy callers; injected stores do not need it.
        _db_path = db_path or self.config.db_path
        self.db_path = Path(_db_path).resolve()

        self.store = store if store is not None else MemoryStore(self.db_path)
        initialize = getattr(self.store, "initialize", None)
        if initialize is not None:
            initialize()

        self.embeddings = (
            embedder
            if embedder is not None
            else create_embedding_engine(
                provider=self.config.embedding.provider,
                model_name=self.config.embedding.model_name,
                dimension=self.config.embedding.dimension,
                cache_size=self.config.embedding.cache_size,
            )
        )

        self.forgetting = ForgettingCurve(self.config.forcing)
        self.retrieval = (
            retrieval
            if retrieval is not None
            else RetrievalEngine(self.store, self.embeddings, self.config.retrieval)
        )
        self.trust_policy = (
            trust_policy
            if trust_policy is not None
            else DefaultTrustPolicy(self.config.trust.minimum_confidence)
        )
        self.evolution_planner = evolution_planner or OfflineEvolutionPlanner()
        self.consolidator = MemoryConsolidator(
            self.store,
            EmbeddingSimilarity(self.embeddings),
            self.config.consolidation,
        )
        self.context_packer = ContextBudgetPacker(
            budget_chars=self.config.retrieval.context_budget_chars,
            reserved_chars=self.config.retrieval.reserved_context_chars,
        )

        # Agent state and active tenant context.
        self.state = AgentState()
        self.namespace: str | None = None
        self.user_id: str | None = None
        self.session_label = ""

    def init_session(
        self,
        label: str = "",
        *,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Start a new session in an isolated namespace."""
        self.session_label = label
        self.namespace = namespace if namespace is not None else user_id
        self.user_id = user_id
        self.state.session_id = self.store.create_session(
            label=label, namespace=self.namespace
        )
        self.state.turn_count = 0
        self.state.session_memories = 0
        self.state.total_memories = self.store.count_memories(namespace=self.namespace)
        self.state.current_context = []

    def end_session(self) -> None:
        """End the current session."""
        if self.state.session_id is not None:
            self.store.end_session(self.state.session_id, namespace=self.namespace)
    def perceive(
        self,
        user_input: str,
        agent_response: str | None = None,
        *,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process one input through extraction, recall, and lifecycle updates."""
        previous_namespace = self.namespace
        if namespace is not None or user_id is not None:
            requested_namespace = namespace if namespace is not None else user_id
            self.namespace = requested_namespace
            if user_id is not None:
                self.user_id = user_id
            if requested_namespace != previous_namespace:
                if self.state.session_id is not None:
                    previous_session_id = self.state.session_id
                    self.store.end_session(
                        previous_session_id,
                        namespace=previous_namespace,
                        commit=False,
                    )
                    self.state.session_id = self.store.create_session(
                        label=self.session_label,
                        namespace=requested_namespace,
                        commit=False,
                    )
                self.state.turn_count = 0
                self.state.session_memories = 0
                self.state.current_context = []
        active_namespace = self.namespace
        active_user_id = self.user_id
        self.state.turn_count += 1
        recollections: list[SearchResult] = []
        new_memories: list[MemoryRecord] = []
        archived = 0
        consolidation_decisions: list[ConsolidationDecision] = []
        recall_packet: RecallPacket | None = None
        decay_applied = False

        # --- 1. EXTRACT and CONSOLIDATE memories from user input ---
        forget_query = extract_forget_query(user_input)
        if forget_query:
            archived += self.consolidator.forget_matching(
                forget_query, namespace=active_namespace, commit=False
            )

        extracted = extract_from_input(user_input)
        for mem in extracted:
            mem.namespace = active_namespace
            if active_user_id is not None:
                mem.metadata = {**mem.metadata, "user_id": active_user_id}
            decision = self.consolidator.consolidate(
                mem, namespace=active_namespace, commit=False
            )
            consolidation_decisions.append(decision)
            if decision.new_memory_id is not None:
                mem.id = decision.new_memory_id
                self._index_stored_memory(mem, namespace=active_namespace)
                new_memories.append(mem)
            if decision.action is ConsolidationAction.UPDATE:
                archived += 1

        # --- 2. RETRIEVE relevant memories and apply trust policy ---
        if self.store.count_memories(namespace=active_namespace) > 0:
            candidate_recollections = self.retrieval.retrieve(
                query=user_input,
                top_k=self.config.retrieval.top_k,
                candidate_k=self.config.retrieval.candidate_k,
                use_mmr=True,
                namespace=active_namespace,
                commit=False,
                record_access=False,
                include_related=True,
            )
            self._apply_trust_policy(candidate_recollections)
            recall_packet = self.context_packer.pack(candidate_recollections)
            recollections = recall_packet.selected

            # Reinforce strength for retrieved memories.
            for r in recollections:
                self.forgetting.reinforce(r.memory)
                if r.memory.id is not None:
                    self.store.update_memory(
                        r.memory, namespace=active_namespace, commit=False
                    )

        # --- 3. STORE the interaction as an episodic memory ---
        if should_remember(user_input, agent_response or ""):
            summary = summarize_interaction(user_input, agent_response or "")
            metadata = {"user_id": active_user_id} if active_user_id is not None else {}
            ep_mem = MemoryRecord(
                content=summary,
                memory_type="episodic",
                importance=0.4,
                tags=["interaction"],
                namespace=active_namespace,
                metadata=metadata,
            )
            ep_id = self._store_memory(ep_mem, namespace=active_namespace)
            new_memories.append(ep_mem)

            # Link to session.
            if self.state.session_id is not None:
                self.store.link_memory_to_session(
                    self.state.session_id,
                    ep_id,
                    turn_index=self.state.turn_count,
                    namespace=active_namespace,
                    commit=False,
                )

        # --- 4. APPLY forgetting decay ---
        if self.state.turn_count % self.config.consolidation.decay_interval == 0:
            self._run_decay_cycle(namespace=active_namespace)
            archived = self.store.archive_below_threshold(
                self.config.forcing.archival_threshold,
                namespace=active_namespace,
                commit=False,
            )
            decay_applied = True

        # --- 5. Update state and return backwards-compatible result ---
        self.store.commit()
        self.state.current_context = [r.memory for r in recollections]
        self.state.total_memories = self.store.count_memories(namespace=active_namespace)
        context_str = self._format_context(recollections)
        evidence = []
        for result in (
            recall_packet.selected + recall_packet.omitted if recall_packet else []
        ):
            if result.evidence is None:
                continue
            if result.relation_evidence:
                evidence_payload = result.evidence.to_dict()
                evidence_payload["relation_evidence"] = result.to_dict()[
                    "relation_evidence"
                ]
                evidence.append(evidence_payload)
            else:
                evidence.append(result.evidence)
        lifecycle = {
            "namespace": active_namespace,
            "user_id": active_user_id,
            "consolidation": consolidation_decisions,
            "decay_applied": decay_applied,
            "archived": archived,
        }

        return {
            "recollections": recollections,
            "recollection_text": context_str,
            "new_memories": new_memories,
            "turn_count": self.state.turn_count,
            "total_memories": self.state.total_memories,
            "archived": archived,
            "recall_packet": recall_packet,
            "consolidation_decisions": consolidation_decisions,
            "evidence": evidence,
            "lifecycle": lifecycle,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def store_memory(
        self, memory: MemoryRecord, *, namespace: str | None = None
    ) -> int:
        """Store a memory in the explicit or active namespace."""
        active_namespace = namespace if namespace is not None else self.namespace
        return self._store_memory(memory, namespace=active_namespace)

    def search_memories(
        self,
        query: str,
        *,
        top_k: int = 5,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Search memories through the trust-aware public agent boundary."""
        active_namespace = namespace if namespace is not None else self.namespace
        results = self.retrieval.retrieve(
            query,
            top_k=top_k,
            memory_type=memory_type,
            use_mmr=True,
            namespace=active_namespace,
            include_related=True,
        )
        self._apply_trust_policy(results)
        packet = self.context_packer.pack(results)
        evidence: list[dict[str, Any]] = []
        for result in packet.selected + packet.omitted:
            item = {
                "id": result.memory.id,
                **(
                    result.evidence.to_dict()
                    if result.evidence is not None
                    else {}
                ),
            }
            if id(result) in packet.reasons:
                item["reason"] = packet.reasons[id(result)]
            evidence.append(item)
        return {
            "namespace": active_namespace,
            "results": [result.to_dict() for result in packet.selected],
            "total": len(packet.selected),
            "selected_ids": packet.selected_ids,
            "dropped_ids": packet.dropped_ids,
            "evidence": evidence,
            "lifecycle": {
                "operation": "search",
                "status": "searched",
                "namespace": active_namespace,
            },
        }

    def evolve_memory(
        self,
        candidate: MemoryRecord,
        neighbors: list[MemoryRecord],
        context: dict[str, Any] | None = None,
        *,
        namespace: str | None = None,
        planner: EvolutionPlannerPort | None = None,
    ) -> EvolutionDecision | None:
        """Plan and apply one explicit, auditable memory evolution."""
        active_namespace = namespace if namespace is not None else self.namespace
        planning_context = dict(context or {})
        planning_context.setdefault("namespace", active_namespace)
        selected_planner = planner or self.evolution_planner
        proposal = selected_planner.propose(candidate, neighbors, planning_context)
        if proposal is None:
            return None
        return self.store.apply_evolution(proposal)

    def forget_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> dict[str, Any]:
        """Archive one memory through the namespace-aware facade."""
        active_namespace = namespace if namespace is not None else self.namespace
        memory = self.store.get_memory(memory_id, namespace=active_namespace)
        if memory is None:
            return {
                "id": memory_id,
                "namespace": active_namespace,
                "status": "not_found",
                "trust": "unknown",
                "reason": "memory not found in namespace",
                "lifecycle": "unchanged",
            }
        policy_evidence = self.trust_policy.evaluate(memory)
        self.store.archive_memory(
            memory_id,
            reason="explicit user request",
            namespace=active_namespace,
        )
        return {
            "id": memory_id,
            "namespace": active_namespace,
            "status": "archived",
            "trust": policy_evidence.trust,
            "reason": (
                "explicit user request; "
                f"{policy_evidence.reason or 'trust policy evaluated'}"
            ),
            "lifecycle": "archived",
        }

    def reinforce_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> dict[str, Any]:
        """Reinforce one memory through the facade."""
        active_namespace = namespace if namespace is not None else self.namespace
        memory = self.store.get_memory(memory_id, namespace=active_namespace)
        if memory is None:
            return {
                "id": memory_id,
                "namespace": active_namespace,
                "status": "not_found",
                "reason": "memory not found in namespace",
                "lifecycle": "unchanged",
            }
        self.forgetting.reinforce(memory)
        self.store.update_memory(memory, namespace=active_namespace)
        return {
            "id": memory_id,
            "namespace": active_namespace,
            "new_strength": round(memory.strength, 3),
            "status": "reinforced",
            "reason": "explicit reinforcement",
            "lifecycle": "reinforced",
        }

    def list_memories(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        """List active memories through the facade."""
        active_namespace = namespace if namespace is not None else self.namespace
        return self.store.get_all_active_memories(namespace=active_namespace)
    def explain_memory(self, memory: MemoryRecord) -> dict[str, Any]:
        """Return trust evidence for a memory through the public facade."""
        return self.trust_policy.evaluate(memory).to_dict()

    def _store_memory(
        self, memory: MemoryRecord, *, namespace: str | None = None
    ) -> int:
        """Store a memory and its embedding."""
        active_namespace = namespace if namespace is not None else memory.namespace
        if active_namespace is None:
            active_namespace = self.namespace
        memory.namespace = active_namespace
        mid = self.store.add_memory(memory, namespace=active_namespace, commit=False)
        memory.id = mid
        self._index_stored_memory(memory, namespace=active_namespace)
        return mid

    def _index_stored_memory(
        self, memory: MemoryRecord, *, namespace: str | None = None
    ) -> None:
        """Store embedding and session accounting for a persisted memory."""
        if memory.id is None:
            return

        active_namespace = namespace if namespace is not None else memory.namespace
        try:
            blob = self.embeddings.encode(memory.content)
            self.store.save_embedding(
                memory.id,
                blob,
                self.embeddings.model_name,
                namespace=active_namespace,
                commit=False,
            )
        except Exception as exc:
            if getattr(self.embeddings, "provider", None) == "deterministic-fallback":
                # Legacy direct EmbeddingEngine() callers retain keyword-only fallback.
                pass
            else:
                raise RuntimeError(
                    "Configured embedding provider failed while indexing a memory"
                ) from exc

        self.state.session_memories += 1

    def _apply_trust_policy(self, results: list[SearchResult]) -> None:
        """Attach policy trust to existing retrieval evidence without duplication."""
        for result in results:
            policy_evidence = self.trust_policy.evaluate(result.memory)
            base_evidence = result.evidence or RetrievalEvidence(
                score=result.score,
                semantic_score=result.semantic_score,
                recency_score=result.recency_score,
                importance_score=result.importance_score,
                strength_score=result.strength_score,
                matched_by=result.matched_by,
            )
            if policy_evidence.reason:
                reason = policy_evidence.reason
            elif policy_evidence.trust != base_evidence.trust:
                reason = f"trust policy classified memory as {policy_evidence.trust}"
            else:
                reason = base_evidence.reason
            result.evidence = replace(
                base_evidence,
                trust=policy_evidence.trust,
                reason=reason,
            )

    def _run_decay_cycle(self, *, namespace: str | None = None) -> None:
        """Apply forgetting curve to active memories in one namespace."""
        memories = self.store.get_all_active_memories(namespace=namespace)
        updates = self.forgetting.decay_all(memories, datetime.now())
        self.store.update_strengths(updates, namespace=namespace, commit=False)

    def _format_context(self, recollections: list[SearchResult]) -> str:
        """Format retrieved memories into a context string for the agent."""
        if not recollections:
            return ""

        lines = ["[Retrieved memories]:\n"]
        for i, r in enumerate(recollections, 1):
            mem = r.memory
            lines.append(
                f"  {i}. [{mem.memory_type}] (importance={mem.importance:.1f}, "
                f"strength={mem.strength:.2f}) {mem.content}"
            )
            if mem.tags:
                tags_str = ", ".join(mem.tags[:3])
                lines.append(f"     tags: {tags_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, *, namespace: str | None = None) -> dict[str, Any]:
        """Get current memory agent statistics for a namespace."""
        active_namespace = namespace if namespace is not None else self.namespace
        all_memories = self.store.get_all_active_memories(namespace=active_namespace)
        type_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        total_importance = 0.0
        archived = self.store.count_memories(
            active_only=False, namespace=active_namespace
        ) - len(all_memories)

        for mem in all_memories:
            type_counts[mem.memory_type] = type_counts.get(mem.memory_type, 0) + 1
            total_importance += mem.importance
            for tag in mem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "namespace": active_namespace,
            "total_active": len(all_memories),
            "archived": max(0, archived),
            "session_turns": self.state.turn_count,
            "type_distribution": dict(sorted(type_counts.items())),
            "tag_distribution": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "avg_importance": round(total_importance / max(len(all_memories), 1), 2),
            "embedding_count": self.store.get_embedding_count(namespace=active_namespace),
            "decay_lifespans_days": self.forgetting.decay_samples(),
            "lifecycle": "active",
        }

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()
