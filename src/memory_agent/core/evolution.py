"""Deterministic, proposal-first memory evolution planning."""

from __future__ import annotations

import math
from typing import Any

from memory_agent.models import EvolutionProposal, MemoryRecord, RetrievalEvidence


class OfflineEvolutionPlanner:
    """Generate conservative supersession proposals without an LLM."""

    minimum_confidence = 0.5

    def propose(
        self,
        candidate: MemoryRecord,
        neighbors: list[MemoryRecord],
        context: dict[str, Any],
    ) -> EvolutionProposal | None:
        """Return a stable proposal only when evidence is explicitly trusted."""
        if context.get("evidence_trust") != "trusted":
            return None
        if candidate.id is None or not candidate.is_active:
            return None
        if candidate.confidence is None:
            return None
        try:
            if not math.isfinite(candidate.confidence) or candidate.confidence < self.minimum_confidence:
                return None
        except (TypeError, ValueError):
            return None

        namespace = context.get("namespace", candidate.namespace)
        if namespace != candidate.namespace:
            return None
        eligible: list[MemoryRecord] = []
        for memory in neighbors:
            if (
                memory.id is None
                or memory.id == candidate.id
                or not memory.is_active
                or memory.namespace != namespace
                or memory.confidence is None
            ):
                continue
            try:
                trusted_confidence = math.isfinite(memory.confidence) and memory.confidence >= self.minimum_confidence
            except (TypeError, ValueError):
                trusted_confidence = False
            if trusted_confidence:
                eligible.append(memory)
        if not eligible:
            return None
        target = min(eligible, key=lambda memory: (memory.id, memory.content))
        return EvolutionProposal(
            candidate_id=candidate.id,
            target_ids=(target.id,),
            action="supersede",
            relation_type="supersedes",
            metadata_patch={"evolution": "accepted", "source": "offline-planner"},
            confidence=float(candidate.confidence),
            actor="offline-planner",
            reason="deterministic candidate supersedes trusted neighbor",
            namespace=namespace,
            evidence=RetrievalEvidence(
                score=float(candidate.confidence),
                matched_by=("offline",),
                trust="trusted",
                reason="trusted evidence and sufficient candidate confidence",
            ),
        )
