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

from datetime import datetime
from pathlib import Path
from typing import Any

from memory_agent.agent.decision import (
    extract_forget_query,
    extract_from_input,
    should_remember,
    summarize_interaction,
)
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.embeddings import EmbeddingEngine
from memory_agent.core.consolidation import (
    ConsolidationAction,
    ConsolidationDecision,
    MemoryConsolidator,
)
from memory_agent.core.context_budget import ContextBudgetPacker, RecallPacket
from memory_agent.core.forgetting import ForgettingCurve
from memory_agent.core.memory_store import MemoryStore
from memory_agent.core.retrieval import RetrievalEngine
from memory_agent.models import AgentState, MemoryRecord, SearchResult

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
            "el usuario prefiere:",
            "al usuario no le gusta:",
            "hecho:",
            "usuario:",
            "respuesta:",
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
    ):
        self.config = config or MemoryAgentConfig.default()

        # Resolve db path
        _db_path = db_path or self.config.db_path
        self.db_path = Path(_db_path).resolve()

        # Initialize components
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

        self.embeddings = EmbeddingEngine(
            model_name=self.config.embedding.model_name,
            cache_size=self.config.embedding.cache_size,
        )

        self.forgetting = ForgettingCurve(self.config.forcing)
        self.retrieval = RetrievalEngine(
            self.store, self.embeddings, self.config.retrieval
        )
        self.consolidator = MemoryConsolidator(
            self.store,
            EmbeddingSimilarity(self.embeddings),
            self.config.consolidation,
        )
        self.context_packer = ContextBudgetPacker(
            budget_chars=self.config.retrieval.context_budget_chars,
            reserved_chars=self.config.retrieval.reserved_context_chars,
        )

        # Agent state
        self.state = AgentState()

    def init_session(self, label: str = "") -> None:
        """Start a new session."""
        self.state.session_id = self.store.create_session(label=label)
        self.state.turn_count = 0
        self.state.session_memories = 0
        self.state.total_memories = self.store.count_memories()
        self.state.current_context = []

    def end_session(self) -> None:
        """End the current session."""
        if self.state.session_id is not None:
            self.store.end_session(self.state.session_id)

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def perceive(
        self, user_input: str, agent_response: str | None = None
    ) -> dict[str, Any]:
        """Process a user input through the full memory cycle.

        Args:
            user_input: The user's message.
            agent_response: Optional agent response (can be set later).

        Returns:
            Dict with:
              - recollections: list of SearchResult
              - new_memories: list of MemoryRecord added
              - turn_count: current turn number
              - total_memories: total stored memories
              - archived: count of memories archived this turn
        """
        self.state.turn_count += 1
        recollections: list[SearchResult] = []
        new_memories: list[MemoryRecord] = []
        archived = 0
        consolidation_decisions: list[ConsolidationDecision] = []
        recall_packet: RecallPacket | None = None

        # --- 1. EXTRACT and CONSOLIDATE memories from user input ---
        forget_query = extract_forget_query(user_input)
        if forget_query:
            archived += self.consolidator.forget_matching(forget_query)

        extracted = extract_from_input(user_input)
        for mem in extracted:
            decision = self.consolidator.consolidate(mem)
            consolidation_decisions.append(decision)
            if decision.new_memory_id is not None:
                self._index_stored_memory(mem)
                new_memories.append(mem)
            if decision.action is ConsolidationAction.UPDATE:
                archived += 1

        # --- 2. RETRIEVE relevant memories ---
        if self.store.count_memories() > 0:
            candidate_recollections = self.retrieval.retrieve(
                query=user_input,
                top_k=self.config.retrieval.top_k,
                candidate_k=self.config.retrieval.candidate_k,
                use_mmr=True,
            )
            recall_packet = self.context_packer.pack(candidate_recollections)
            recollections = recall_packet.selected

            # Reinforce strength for retrieved memories
            for r in recollections:
                self.forgetting.reinforce(r.memory)
                if r.memory.id is not None:
                    self.store.update_memory(r.memory, commit=False)

        # --- 3. STORE the interaction as an episodic memory ---
        if should_remember(user_input, agent_response or ""):
            summary = summarize_interaction(user_input, agent_response or "")
            ep_mem = MemoryRecord(
                content=summary,
                memory_type="episodic",
                importance=0.4,
                tags=["interaction"],
            )
            ep_id = self._store_memory(ep_mem)
            new_memories.append(ep_mem)

            # Link to session
            if self.state.session_id is not None:
                self.store.link_memory_to_session(
                    self.state.session_id,
                    ep_id,
                    turn_index=self.state.turn_count,
                    commit=False,
                )

        # --- 4. APPLY forgetting decay ---
        if self.state.turn_count % self.config.consolidation.decay_interval == 0:
            self._run_decay_cycle()
            archived = self.store.archive_below_threshold(
                self.config.forcing.archival_threshold, commit=False
            )

        # --- 5. Update state ---
        self.store.conn.commit()
        self.state.current_context = [r.memory for r in recollections]
        self.state.total_memories = self.store.count_memories()

        # Build memory context string
        context_str = self._format_context(recollections)

        return {
            "recollections": recollections,
            "recollection_text": context_str,
            "new_memories": new_memories,
            "turn_count": self.state.turn_count,
            "total_memories": self.state.total_memories,
            "archived": archived,
            "recall_packet": recall_packet,
            "consolidation_decisions": consolidation_decisions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def store_memory(self, memory: MemoryRecord) -> int:
        """Store a memory with its embedding (public wrapper around _store_memory).

        Args:
            memory: The MemoryRecord to store.

        Returns:
            The new memory ID.
        """
        return self._store_memory(memory)

    def _store_memory(self, memory: MemoryRecord) -> int:
        """Store a memory and its embedding."""
        mid = self.store.add_memory(memory, commit=False)
        self._index_stored_memory(memory)
        return mid

    def _index_stored_memory(self, memory: MemoryRecord) -> None:
        """Store embedding and session accounting for an already-persisted memory."""
        if memory.id is None:
            return

        try:
            blob = self.embeddings.encode(memory.content)
            self.store.save_embedding(
                memory.id, blob, self.embeddings.model_name, commit=False
            )
        except Exception:
            # If embedding fails, store without it (will fall back to keyword search)
            pass

        self.state.session_memories += 1

    def _run_decay_cycle(self) -> None:
        """Apply forgetting curve to all active memories."""
        memories = self.store.get_all_active_memories()
        updates = self.forgetting.decay_all(memories, datetime.now())
        self.store.update_strengths(updates, commit=False)

    def _format_context(self, recollections: list[SearchResult]) -> str:
        """Format retrieved memories into a context string for the agent."""
        if not recollections:
            return ""

        lines = ["[Recuerdos recuperados]:\n"]
        for i, r in enumerate(recollections, 1):
            mem = r.memory
            lines.append(
                f"  {i}. [{mem.memory_type}] (importancia={mem.importance:.1f}, "
                f"fuerza={mem.strength:.2f}) {mem.content}"
            )
            if mem.tags:
                tags_str = ", ".join(mem.tags[:3])
                lines.append(f"     tags: {tags_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get current memory agent statistics."""
        all_memories = self.store.get_all_active_memories()
        type_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        total_importance = 0.0
        archived = self.store.count_memories(active_only=False) - len(all_memories)

        for mem in all_memories:
            type_counts[mem.memory_type] = type_counts.get(mem.memory_type, 0) + 1
            total_importance += mem.importance
            for tag in mem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_active": len(all_memories),
            "archived": max(0, archived),
            "session_turns": self.state.turn_count,
            "type_distribution": dict(sorted(type_counts.items())),
            "tag_distribution": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "avg_importance": round(total_importance / max(len(all_memories), 1), 2),
            "embedding_count": self.store.get_embedding_count(),
            "decay_lifespans_days": self.forgetting.decay_samples(),
        }

    def close(self) -> None:
        self.store.close()
