"""Context-budget-aware recall packing."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory_agent.models import SearchResult


@dataclass
class RecallPacket:
    """Selected and omitted recollections for a bounded context window."""

    selected: list[SearchResult]
    omitted: list[SearchResult]
    budget_chars: int
    reserved_chars: int
    used_chars: int
    reasons: dict[int, str] = field(default_factory=dict)

    @property
    def available_chars(self) -> int:
        return max(0, self.budget_chars - self.reserved_chars)

    @property
    def selected_ids(self) -> list[int]:
        """Memory IDs selected for the context packet."""
        return [result.memory.id for result in self.selected if result.memory.id is not None]

    @property
    def dropped_ids(self) -> list[int]:
        """Memory IDs omitted from the context packet."""
        return [result.memory.id for result in self.omitted if result.memory.id is not None]

    @property
    def limit(self) -> int:
        """Configured total character limit, including reserved space."""
        return self.budget_chars


class ContextBudgetPacker:
    """Packs ranked memories into a character budget."""

    def __init__(self, budget_chars: int, reserved_chars: int = 0) -> None:
        self.budget_chars = max(0, budget_chars)
        self.reserved_chars = max(0, reserved_chars)

    def pack(self, results: list[SearchResult]) -> RecallPacket:
        available = max(0, self.budget_chars - self.reserved_chars)
        selected: list[SearchResult] = []
        omitted: list[SearchResult] = []
        reasons: dict[int, str] = {}
        used = 0

        # Trust filtering is deliberately performed before ranking and budget
        # accounting, so untrusted memories cannot consume context capacity.
        ranked_input: list[SearchResult] = []
        for result in results:
            if result.trust == "untrusted":
                omitted.append(result)
                reasons[id(result)] = (
                    f"omitted: trust={result.trust}; {result.reason or 'not trusted'}"
                )
            else:
                ranked_input.append(result)

        ranked = sorted(
            ranked_input,
            key=lambda r: (
                r.score / max(r.estimated_chars, 1),
                r.score,
                r.memory.importance,
            ),
            reverse=True,
        )

        for result in ranked:
            cost = result.estimated_chars
            result_key = id(result)
            if cost > available:
                omitted.append(result)
                reasons[result_key] = (
                    f"omitted: too large for available budget "
                    f"({cost} chars > {available} chars)"
                )
                continue
            if used + cost <= available:
                selected.append(result)
                used += cost
                reasons[result_key] = (
                    f"selected: score={result.score:.3f}, "
                    f"cost={cost}, used={used}/{available}"
                )
            else:
                omitted.append(result)
                reasons[result_key] = (
                    f"omitted: remaining budget too small "
                    f"({available - used} chars left, needs {cost})"
                )

        return RecallPacket(
            selected=selected,
            omitted=omitted,
            budget_chars=self.budget_chars,
            reserved_chars=self.reserved_chars,
            used_chars=used,
            reasons=reasons,
        )
