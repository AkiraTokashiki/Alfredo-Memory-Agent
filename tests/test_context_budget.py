"""Tests for context-budget recall packing."""

from __future__ import annotations

from memory_agent.core.config import ConsolidationConfig, RetrievalConfig
from memory_agent.core.context_budget import ContextBudgetPacker
from memory_agent.models import MemoryRecord, SearchResult


def test_retrieval_config_exposes_context_budget_settings():
    config = RetrievalConfig()

    assert config.candidate_k >= config.top_k
    assert config.context_budget_chars > 0
    assert config.reserved_context_chars >= 0


def test_consolidation_config_exposes_similarity_thresholds():
    config = ConsolidationConfig()

    assert 0.0 < config.duplicate_similarity_threshold <= 1.0
    assert 0.0 < config.supersede_similarity_threshold <= 1.0
    assert config.explicit_forget_min_score >= 0.0


def test_search_result_estimated_chars_is_memory_content_length():
    memory = MemoryRecord(content="El usuario prefiere: Rust", memory_type="preference")
    result = SearchResult(memory=memory, score=0.9)

    assert result.estimated_chars == len(memory.content)



def _result(content: str, score: float, importance: float = 0.5) -> SearchResult:
    memory = MemoryRecord(content=content, memory_type="preference", importance=importance)
    return SearchResult(memory=memory, score=score, importance_score=importance)


def test_context_budget_selects_memories_that_fit():
    packer = ContextBudgetPacker(budget_chars=80, reserved_chars=10)
    results = [
        _result("critical short preference", 0.95, 0.9),
        _result("x" * 200, 0.99, 1.0),
        _result("minor memory", 0.2, 0.2),
    ]

    packet = packer.pack(results)

    assert [r.memory.content for r in packet.selected] == [
        "critical short preference",
        "minor memory",
    ]
    assert packet.omitted[0].memory.content == "x" * 200
    assert packet.used_chars <= packet.available_chars
    assert "selected" in packet.reasons[id(packet.selected[0])]
    assert "too large" in packet.reasons[id(packet.omitted[0])]


def test_context_budget_returns_empty_when_reserved_exhausts_budget():
    packer = ContextBudgetPacker(budget_chars=10, reserved_chars=10)
    packet = packer.pack([_result("important", 1.0, 1.0)])

    assert packet.selected == []
    assert len(packet.omitted) == 1
    assert packet.available_chars == 0