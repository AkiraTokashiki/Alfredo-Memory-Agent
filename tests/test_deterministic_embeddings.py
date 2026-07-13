"""Contract tests for deterministic offline embeddings."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine
from memory_agent.core.embeddings import EmbeddingEngine, create_embedding_engine


def test_equal_inputs_produce_identical_normalized_vectors() -> None:
    engine = DeterministicEmbeddingEngine(dimension=32)

    first = engine.decode_vector(engine.encode("I prefer Python"))
    second = engine.decode_vector(engine.encode("I prefer Python"))

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.shape == (32,)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_different_inputs_produce_bounded_cosine_compatible_vectors() -> None:
    engine = DeterministicEmbeddingEngine(dimension=64)

    left = engine.decode_vector(engine.encode("I prefer Python"))
    right = engine.decode_vector(engine.encode("The weather is sunny"))
    similarity = engine.cosine_similarity(left, right)

    assert not np.array_equal(left, right)
    assert -1.0 <= similarity <= 1.0
    assert np.isclose(np.linalg.norm(right), 1.0)


def test_dimension_is_configurable_and_model_name_is_explicit() -> None:
    engine = DeterministicEmbeddingEngine(dimension=7)

    vector = engine.decode_vector(engine.encode("offline"))

    assert vector.shape == (7,)
    assert engine.dimension == 7
    assert engine.model_name == "deterministic-hashed-token"

def test_empty_or_punctuation_input_still_has_normalized_vector() -> None:
    engine = DeterministicEmbeddingEngine(dimension=32)

    vector = engine.decode_vector(engine.encode("... !!!"))

    assert np.linalg.norm(vector) > 0
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_explicit_production_provider_does_not_fallback_when_dependency_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        create_embedding_engine(provider="sentence-transformers")

def test_semantic_vectors_persist_provider_provenance() -> None:
    engine = EmbeddingEngine(provider="sentence-transformers")

    class FakeModel:
        def encode(self, text: str, **kwargs):
            return np.ones(4, dtype=np.float32)

    engine._model = FakeModel()
    engine._dimension = 4
    vector = engine.decode_vector(engine.encode("semantic"))

    assert getattr(vector, "provenance", None) == "sentence-transformers"

def test_legacy_transformer_override_uses_semantic_provenance(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS", "1")
    engine = EmbeddingEngine()

    class FakeModel:
        def encode(self, text: str, **kwargs):
            return np.ones(4, dtype=np.float32)

    engine._model = FakeModel()
    engine._dimension = 4
    vector = engine.decode_vector(engine.encode("override"))

    assert engine.provider == "sentence-transformers"
    assert engine.model_name == "all-MiniLM-L6-v2"
    assert getattr(vector, "provenance", None) == "sentence-transformers"

def test_legacy_mode_is_fixed_when_env_changes_after_construction(monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS", raising=False)
    engine = EmbeddingEngine()
    engine._dimension = 4

    class ExplodingModel:
        def encode(self, text: str, **kwargs):
            raise AssertionError("legacy mode must not call the transformer")

    engine._model = ExplodingModel()
    monkeypatch.setenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS", "1")
    vector = engine.decode_vector(engine.encode("fixed mode"))

    assert engine.provider == "deterministic-fallback"
    assert engine.model_name == "deterministic-fallback-v1"
    assert getattr(vector, "provenance", None) is None
