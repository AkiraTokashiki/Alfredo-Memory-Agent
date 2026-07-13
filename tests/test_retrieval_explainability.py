"""Focused tests for explainable retrieval and deterministic packing metadata."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memory_agent.core.config import RetrievalConfig
from memory_agent.core.context_budget import ContextBudgetPacker
from memory_agent.core.memory_store import MemoryStore
from memory_agent.core.retrieval import RetrievalEngine
from memory_agent.models import MemoryRecord, SearchResult


class FakeEmbeddings:
    model_name = "fake"

    def encode(self, text: str) -> bytes:
        return pickle.dumps(np.asarray([1.0, 0.0], dtype=np.float32))

    def decode_vector(self, blob: bytes) -> np.ndarray:
        return pickle.loads(blob)

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


@pytest.fixture
def store() -> MemoryStore:
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    value = MemoryStore(db.name)
    value.initialize()
    yield value
    value.close()
    Path(db.name).unlink(missing_ok=True)


def _add(store: MemoryStore, content: str, *, namespace: str, confidence: float | None = 0.9) -> int:
    memory = MemoryRecord(content=content, namespace=namespace, confidence=confidence)
    memory_id = store.add_memory(memory, namespace=namespace)
    store.save_embedding(memory_id, pickle.dumps(np.asarray([1.0, 0.0], dtype=np.float32)), "fake", namespace=namespace)
    return memory_id


def test_retrieval_result_exposes_component_evidence_and_reason(store: MemoryStore):
    memory_id = _add(store, "trusted preference", namespace="tenant-a")
    engine = RetrievalEngine(store, FakeEmbeddings(), RetrievalConfig(top_k=1, candidate_k=1))

    result = engine.retrieve("preference", namespace="tenant-a", use_mmr=False)[0]

    assert result.memory.id == memory_id
    assert result.evidence is not None
    assert result.evidence.score == pytest.approx(result.score)
    assert result.evidence.semantic_score == pytest.approx(result.semantic_score)
    assert result.evidence.recency_score == pytest.approx(result.recency_score)
    assert result.evidence.importance_score == pytest.approx(result.importance_score)
    assert result.evidence.strength_score == pytest.approx(result.strength_score)
    assert result.evidence.matched_by == ("semantic", "recency", "importance", "strength")
    assert result.evidence.trust == "trusted"
    assert result.evidence.reason
    assert result.matched_by == result.evidence.matched_by
    assert result.trust == "trusted"
    assert result.reason == result.evidence.reason


def test_namespace_filtering_happens_before_ranking(store: MemoryStore):
    tenant_a_id = _add(store, "same query in a", namespace="tenant-a")
    _add(store, "same query in b", namespace="tenant-b")
    engine = RetrievalEngine(store, FakeEmbeddings(), RetrievalConfig(top_k=10, candidate_k=10))

    results = engine.retrieve("same query", namespace="tenant-a", use_mmr=False)

    assert [result.memory.id for result in results] == [tenant_a_id]
    assert all(result.memory.namespace == "tenant-a" for result in results)


def test_access_tracking_updates_only_retrieved_namespace(store: MemoryStore):
    tenant_a_id = _add(store, "tenant a preference", namespace="tenant-a")
    tenant_b_id = _add(store, "tenant b preference", namespace="tenant-b")
    engine = RetrievalEngine(
        store, FakeEmbeddings(), RetrievalConfig(top_k=1, candidate_k=1)
    )
    traces: list[str] = []
    store.conn.set_trace_callback(traces.append)

    engine.retrieve("preference", namespace="tenant-a", use_mmr=False)

    assert any(
        "UPDATE memories" in query and "AND namespace" in query
        for query in traces
    )
    assert store.get_memory(tenant_a_id, namespace="tenant-a").access_count == 1
    assert store.get_memory(tenant_b_id, namespace="tenant-b").access_count == 0


def test_context_packet_reports_exact_ids_and_budget_accounting():
    selected = SearchResult(memory=MemoryRecord(id=11, content="abc"), score=0.9)
    dropped = SearchResult(memory=MemoryRecord(id=12, content="x" * 20), score=0.8)
    packet = ContextBudgetPacker(budget_chars=10, reserved_chars=2).pack([selected, dropped])

    assert packet.selected == [selected]
    assert packet.omitted == [dropped]
    assert packet.selected_ids == [11]
    assert packet.dropped_ids == [12]
    assert packet.used_chars == 3
    assert packet.reserved_chars == 2
    assert packet.limit == 10
    assert packet.available_chars == 8


def test_context_packet_filters_untrusted_results_before_packing():
    trusted = SearchResult(
        memory=MemoryRecord(id=21, content="trusted"),
        score=0.1,
        evidence=None,
    )
    untrusted = SearchResult(
        memory=MemoryRecord(id=22, content="untrusted"),
        score=1.0,
        evidence=None,
    )
    from memory_agent.models import RetrievalEvidence

    trusted.evidence = RetrievalEvidence(trust="trusted", reason="ok")
    untrusted.evidence = RetrievalEvidence(trust="untrusted", reason="low confidence")

    packet = ContextBudgetPacker(budget_chars=100).pack([trusted, untrusted])

    assert [result.memory.id for result in packet.selected] == [21]
    assert [result.memory.id for result in packet.omitted] == [22]
    assert "trust" in packet.reasons[id(untrusted)]
