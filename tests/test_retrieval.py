"""Tests for retrieval engine."""

from __future__ import annotations
import tempfile

import pickle
from pathlib import Path

import numpy as np
import pytest

from memory_agent.core.config import RetrievalConfig
from memory_agent.core.embeddings import EmbeddingEngine
from memory_agent.core.memory_store import MemoryStore
from memory_agent.core.retrieval import RetrievalEngine
from memory_agent.models import MemoryRecord

_EMBEDDER: EmbeddingEngine | None = None


def _get_emb() -> EmbeddingEngine:
    """Lazy singleton embedder to avoid model re-initialization."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = EmbeddingEngine()
    return _EMBEDDER


def _seed(store: MemoryStore, texts: list[str], **kw) -> list[int]:
    """Batch-add memories with embeddings. Returns list of memory IDs."""
    emb = _get_emb()
    vectors = emb.encode_multiple(texts)
    ids = []
    for text, vec in zip(texts, vectors, strict=False):
        mem = MemoryRecord(content=text, **kw)
        mid = store.add_memory(mem)
        store.save_embedding(mid, vec, emb.model_name)
        ids.append(mid)
    return ids


@pytest.fixture
def store() -> MemoryStore:
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    s = MemoryStore(db.name)
    s.initialize()
    yield s
    s.close()
    Path(db.name).unlink(missing_ok=True)


@pytest.fixture
def embeddings() -> EmbeddingEngine:
    return _get_emb()


@pytest.fixture
def engine(store: MemoryStore, embeddings: EmbeddingEngine) -> RetrievalEngine:
    return RetrievalEngine(store, embeddings)


class TestRetrievalEngine:
    def test_empty_store_returns_empty(self, engine: RetrievalEngine):
        results = engine.retrieve("anything")
        assert results == []

    def test_semantic_retrieval(self, engine: RetrievalEngine, store: MemoryStore):
        _seed(store, [
            "I like programming in Python",
            "The weather is sunny today",
            "Python is my favorite language",
        ])

        results = engine.retrieve("Python programming language")
        assert len(results) >= 1
        assert "Python" in results[0].memory.content

    def test_rejects_stored_embedding_from_different_model(
        self, store: MemoryStore, embeddings: EmbeddingEngine
    ):
        ids = _seed(store, ["I like coffee"])
        stored = store.get_embedding(ids[0])
        assert stored is not None
        store.save_embedding(ids[0], stored[0], "different-production-model")

        with pytest.raises(ValueError, match="Embedding model mismatch"):
            RetrievalEngine(store, embeddings).retrieve("coffee")

    def test_rejects_stored_embedding_with_incompatible_dimension(
        self, store: MemoryStore, embeddings: EmbeddingEngine
    ):
        ids = _seed(store, ["I like coffee"])
        store.save_embedding(
            ids[0], pickle.dumps(np.zeros(2, dtype=np.float32)), embeddings.model_name
        )

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            RetrievalEngine(store, embeddings).retrieve("coffee")


    def test_rejects_legacy_fallback_blob_labeled_as_semantic_model(
        self, store: MemoryStore
    ):
        legacy = EmbeddingEngine()
        legacy._dimension = 12
        memory_id = store.add_memory(MemoryRecord(content="legacy memory"))
        store.save_embedding(
            memory_id,
            legacy.encode("legacy memory"),
            "all-MiniLM-L6-v2",
        )

        semantic = EmbeddingEngine(provider="sentence-transformers")

        class FakeModel:
            def get_embedding_dimension(self) -> int:
                return 12

            def encode(self, text: str, **kwargs):
                return np.ones(12, dtype=np.float32)

        semantic._model = FakeModel()
        semantic._dimension = 12

        with pytest.raises(ValueError, match="Embedding provenance mismatch"):
            RetrievalEngine(store, semantic).retrieve("legacy")
    def test_importance_boosts_score(self, store: MemoryStore, embeddings: EmbeddingEngine):
        """More important memories should rank higher with similar content."""
        _seed(store, ["I like coffee"], importance=0.3)
        # Add same text with high importance using same vector
        emb = _get_emb()
        vec = emb.encode("I like coffee")
        mid = store.add_memory(MemoryRecord(content="I like coffee", importance=0.9))
        store.save_embedding(mid, vec, emb.model_name)

        engine = RetrievalEngine(store, embeddings)
        results = engine.retrieve("cafe", use_mmr=False)
        assert len(results) == 2
        assert results[0].memory.importance == 0.9

    def test_memory_type_filter(self, store: MemoryStore, embeddings: EmbeddingEngine):
        _seed(store, ["Python is great"], memory_type="semantic")
        _seed(store, ["The user said they use VS Code"], memory_type="episodic")
        engine = RetrievalEngine(store, embeddings)

        eps = engine.retrieve("codigo", memory_type="episodic")
        assert len(eps) >= 1
        assert all(r.memory.memory_type == "episodic" for r in eps)

    def test_min_score_filter(self, store: MemoryStore, embeddings: EmbeddingEngine):
        _seed(store, ["programacion", "recetas de cocina"])
        engine = RetrievalEngine(store, embeddings, RetrievalConfig(min_score=0.5))

        results = engine.retrieve("programacion", min_score=0.5)
        assert any("programacion" in r.memory.content for r in results)

    def test_top_k_limit(self, store: MemoryStore, embeddings: EmbeddingEngine):
        texts = [f"memory number {i}" for i in range(20)]
        _seed(store, texts, importance=0.5)
        engine = RetrievalEngine(store, embeddings)

        results = engine.retrieve("memory", top_k=5)
        assert len(results) <= 5

    def test_mmr_diversity(self, store: MemoryStore, embeddings: EmbeddingEngine):
        """MMR should produce diverse results."""
        _seed(store, [
            "Python es un lenguaje de programacion",
            "Python es un lenguaje muy popular",
            "Python se usa para data science",
            "JavaScript es para frontend",
            "Rust es para sistemas",
        ])
        engine = RetrievalEngine(store, embeddings)

        no_mmr = engine.retrieve("lenguaje programacion Python", top_k=3, use_mmr=False)
        python_no = sum(1 for r in no_mmr if "Python" in r.memory.content)

        with_mmr = engine.retrieve(
            "lenguaje programacion Python", top_k=3, use_mmr=True
        )
        python_yes = sum(1 for r in with_mmr if "Python" in r.memory.content)

        assert python_yes <= python_no

    def test_retrieval_updates_access_count(self, store: MemoryStore, embeddings: EmbeddingEngine):
        mid = _seed(store, ["algo importante"], importance=0.9)[0]
        engine = RetrievalEngine(store, embeddings)

        initial = store.get_memory(mid)
        assert initial is not None

        engine.retrieve("importante")
        after = store.get_memory(mid)
        assert after is not None
        assert after.access_count > initial.access_count
