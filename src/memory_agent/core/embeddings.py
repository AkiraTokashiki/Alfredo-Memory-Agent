"""Embedding engine using sentence-transformers."""

from __future__ import annotations

import hashlib
import os
import pickle
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


SUPPORTED_PROVIDERS = {"sentence-transformers", "sentence_transformers", "deterministic"}

def _require_sentence_transformers() -> None:
    """Fail at provider selection when the optional production dependency is absent."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for the configured production "
            "provider; install alfredo-memory-agent[semantic] or select "
            "provider='deterministic'"
        ) from exc


class _ProvenanceVector(np.ndarray):
    """Numpy vector carrying the provider provenance through pickle storage."""

    def __new__(cls, values: np.ndarray, provenance: str):
        vector = np.asarray(values, dtype=np.float32).view(cls)
        vector.provenance = provenance
        return vector

    def __array_finalize__(self, source) -> None:
        if source is not None:
            self.provenance = getattr(source, "provenance", None)

    def __reduce__(self):
        constructor, args, state = super().__reduce__()
        return constructor, args, state + (self.provenance,)

    def __setstate__(self, state) -> None:
        self.provenance = state[-1]
        super().__setstate__(state[:-1])


def _serialize_vector(values: np.ndarray, provenance: str) -> bytes:
    return pickle.dumps(
        {"vector": np.asarray(values, dtype=np.float32), "provenance": provenance}
    )


def create_embedding_engine(
    *,
    provider: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
    dimension: int = 384,
    cache_size: int = 1024,
) -> EmbeddingEngine | DeterministicEmbeddingEngine:
    """Create the explicitly selected embedding provider.

    ``sentence-transformers`` remains the production provider; deterministic
    embeddings are selected only when callers request ``deterministic``.
    """
    if provider == "deterministic":
        return DeterministicEmbeddingEngine(dimension=dimension, cache_size=cache_size)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    _require_sentence_transformers()
    return EmbeddingEngine(
        model_name=model_name,
        cache_size=cache_size,
        provider=provider,
    )


class EmbeddingEngine:
    """Semantic embedding engine wrapping sentence-transformers.

    Encodes text into dense vectors for semantic similarity search.
    Includes LRU cache for frequently-embedded texts.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 1024,
        *,
        dimension: int | None = None,
        provider: str | None = None,
    ):
        effective_provider = provider or "sentence-transformers"
        if effective_provider not in {"sentence-transformers", "sentence_transformers"}:
            raise ValueError(
                "EmbeddingEngine only supports the sentence-transformers provider; "
                "use create_embedding_engine(provider='deterministic') for offline mode"
            )
        if provider is not None:
            _require_sentence_transformers()
        legacy_requested = provider is None
        legacy_transformer = legacy_requested and (
            os.getenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS") == "1"
        )
        self._legacy_fallback = legacy_requested and not legacy_transformer
        self.model_name = (
            model_name if legacy_transformer else "deterministic-fallback-v1"
            if legacy_requested
            else model_name
        )
        self.provider = (
            "sentence-transformers"
            if legacy_transformer
            else "deterministic-fallback"
            if legacy_requested
            else effective_provider
        )
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = dimension
        self._encode = lru_cache(maxsize=cache_size)(self._encode_uncached)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the configured "
                    "production provider; install alfredo-memory-agent[semantic] "
                    "or select provider='deterministic'"
                ) from exc

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model  # trigger lazy load
        assert self._dimension is not None
        return self._dimension

    def _encode_uncached(self, text: str) -> bytes:
        """Encode a single text and return serialized numpy array."""
        if self._legacy_fallback:
            return pickle.dumps(self._fallback_vector(text))
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return _serialize_vector(vec, "sentence-transformers")

    def encode(self, text: str) -> bytes:
        """Encode text to a pickle-dumped float32 numpy vector."""
        return self._encode(text)

    def encode_multiple(self, texts: list[str]) -> list[bytes]:
        """Encode multiple texts efficiently (no caching)."""
        if self._legacy_fallback:
            return [pickle.dumps(self._fallback_vector(text)) for text in texts]
        vectors = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return [_serialize_vector(vector, "sentence-transformers") for vector in vectors]

    def _fallback_vector(self, text: str) -> np.ndarray:
        """Deterministic local vector used when sentence-transformers is unavailable."""
        dimension = self._dimension or 384
        self._dimension = dimension
        seed_model = "all-MiniLM-L6-v2" if self._legacy_fallback else self.model_name
        seed = hashlib.sha256(f"{seed_model}\0{text}".encode("utf-8")).digest()
        values = np.empty(dimension, dtype=np.float32)
        counter = 0
        offset = 0
        while offset < dimension:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for byte in block:
                if offset >= dimension:
                    break
                values[offset] = (byte / 127.5) - 1.0
                offset += 1
            counter += 1
        norm = np.linalg.norm(values)
        if norm > 0:
            values /= norm
        return values.astype(np.float32)

    def decode_vector(self, blob: bytes) -> np.ndarray:
        """Deserialize a vector produced by this engine, including provenance."""
        payload = pickle.loads(blob)  # noqa: S301 — safe, our own data
        if isinstance(payload, dict) and "vector" in payload:
            return _ProvenanceVector(payload["vector"], payload["provenance"])
        return payload

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def similarity_between(self, text_a: str, text_b: str) -> float:
        """Direct similarity between two texts."""
        vec_a = self.decode_vector(self.encode(text_a))
        vec_b = self.decode_vector(self.encode(text_b))
        return self.cosine_similarity(vec_a, vec_b)

    def query_similarity(
        self, query_vec: np.ndarray, memory_vectors: list[tuple[int, bytes]]
    ) -> list[tuple[int, float]]:
        """Return (memory_id, similarity) sorted desc for all candidates."""
        results: list[tuple[int, float]] = []
        for mem_id, blob in memory_vectors:
            mem_vec = self.decode_vector(blob)
            sim = self.cosine_similarity(query_vec, mem_vec)
            results.append((mem_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def batch_similarity(
        self, query_vec: np.ndarray, mem_ids: list[int], vectors: list[np.ndarray]
    ) -> dict[int, float]:
        """Batch cosine similarity. Returns {mem_id: similarity}."""
        results: dict[int, float] = {}
        for mem_id, vec in zip(mem_ids, vectors, strict=False):
            results[mem_id] = self.cosine_similarity(query_vec, vec)
        return results

    def clear_cache(self) -> None:
        self._encode.cache_clear()
