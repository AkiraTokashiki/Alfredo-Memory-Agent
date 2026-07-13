"""Dependency-free deterministic embeddings for offline first-run flows."""

from __future__ import annotations

import hashlib
import pickle
import re
from functools import lru_cache

import numpy as np


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class DeterministicEmbeddingEngine:
    """Hash tokens into normalized vectors without model files or network access.

    The provider intentionally uses only the Python standard library and numpy.
    Hashing each token (rather than the whole input) preserves useful similarity
    for shared terms while keeping output reproducible across processes.
    """

    model_name = "deterministic-hashed-token"

    def __init__(self, dimension: int = 384, cache_size: int = 1024) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._encode = lru_cache(maxsize=cache_size)(self._encode_uncached)

    @property
    def dimension(self) -> int:
        """Number of components emitted for each input."""
        return self._dimension

    def _encode_uncached(self, text: str) -> bytes:
        tokens = _TOKEN_RE.findall(text.casefold()) or ["<empty>"]
        values = np.zeros(self._dimension, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign

            # A second independent bucket reduces the impact of collisions in
            # small configured dimensions while preserving determinism.
            index2 = int.from_bytes(digest[9:17], "big") % self._dimension
            values[index2] += -sign * 0.5

        norm = float(np.linalg.norm(values))
        if norm > 0:
            values /= norm
        return pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)

    def encode(self, text: str) -> bytes:
        """Encode one text as a pickled float32 vector."""
        return self._encode(text)

    def encode_multiple(self, texts: list[str]) -> list[bytes]:
        """Encode a batch while retaining the same deterministic semantics."""
        return [self.encode(text) for text in texts]

    def decode_vector(self, blob: bytes) -> np.ndarray:
        """Deserialize a vector produced by this provider."""
        return pickle.loads(blob)  # noqa: S301 — safe, our own data

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return bounded cosine similarity, including a safe zero-vector case."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    def similarity_between(self, text_a: str, text_b: str) -> float:
        """Compare two texts directly."""
        return self.cosine_similarity(
            self.decode_vector(self.encode(text_a)),
            self.decode_vector(self.encode(text_b)),
        )

    def query_similarity(
        self, query_vec: np.ndarray, memory_vectors: list[tuple[int, bytes]]
    ) -> list[tuple[int, float]]:
        """Return memory IDs sorted by descending cosine similarity."""
        results = [
            (memory_id, self.cosine_similarity(query_vec, self.decode_vector(blob)))
            for memory_id, blob in memory_vectors
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def batch_similarity(
        self, query_vec: np.ndarray, mem_ids: list[int], vectors: list[np.ndarray]
    ) -> dict[int, float]:
        """Compute cosine similarity for each memory vector."""
        return {
            memory_id: self.cosine_similarity(query_vec, vector)
            for memory_id, vector in zip(mem_ids, vectors, strict=False)
        }

    def clear_cache(self) -> None:
        """Drop cached encodings."""
        self._encode.cache_clear()


# Provider is a useful name for callers that do not need the engine suffix.
DeterministicEmbeddingProvider = DeterministicEmbeddingEngine
