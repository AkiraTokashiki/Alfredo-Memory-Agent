"""Configuration for MemoryAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from memory_agent.core.paths import default_memory_db_path


@dataclass
class ForgettingConfig:
    """Ebbinghaus forgetting curve parameters."""

    # Base decay hours for different importance levels
    decay_hours_high: float = 720.0   # importance >= 0.8: ~30 days
    decay_hours_medium: float = 168.0  # importance >= 0.5: ~7 days
    decay_hours_low: float = 24.0     # importance < 0.5: ~1 day

    # How much retrieval reinforces strength
    reinforcement_boost: float = 0.15

    # Threshold below which memories are archived
    archival_threshold: float = 0.05

    # Importance thresholds
    importance_high: float = 0.8
    importance_medium: float = 0.5

    # Max strength
    max_strength: float = 1.0


@dataclass
class RetrievalConfig:
    """Retrieval scoring weights."""

    # Weight for semantic similarity (cosine distance)
    w_semantic: float = 0.40

    # Weight for recency (1 / (1 + hours))
    w_recency: float = 0.20

    # Weight for importance
    w_importance: float = 0.20

    # Weight for recall strength
    w_strength: float = 0.20

    # Maximum memories to retrieve
    top_k: int = 10

    # Candidate pool size before context-budget packing
    candidate_k: int = 20

    # Character budget for formatted recollections
    context_budget_chars: int = 2400

    # Reserved characters for the current prompt and instructions
    reserved_context_chars: int = 600

    # MMR diversity lambda (0 = pure relevance, 1 = pure diversity)
    mmr_lambda: float = 0.5

    # Minimum score to include (filters noise)
    min_score: float = 0.05


@dataclass
class ConsolidationConfig:
    """Memory consolidation parameters."""

    # After this many turns, consolidate short-term → long-term
    consolidation_interval: int = 5

    # Minimum importance to auto-consolidate
    auto_consolidate_threshold: float = 0.3

    # How often (in turns) to run full forgetting decay
    decay_interval: int = 3

    # Max memories to keep in short-term working context
    working_context_size: int = 15

    # Semantic threshold for treating a candidate as duplicate
    duplicate_similarity_threshold: float = 0.88

    # Semantic threshold for detecting same-topic preference replacement
    supersede_similarity_threshold: float = 0.55

    # Minimum score for explicit forget matches
    explicit_forget_min_score: float = 0.35


@dataclass
class EmbeddingConfig:
    """Embedding provider and vector configuration."""

    # Production model provider. Offline callers must opt into deterministic.
    provider: Literal["sentence-transformers", "sentence_transformers", "deterministic"] = (
        "sentence-transformers"
    )

    # Model name from sentence-transformers (ignored by deterministic provider).
    model_name: str = "all-MiniLM-L6-v2"

    # Dimension of the model output vectors.
    dimension: int = 384

    # Cache size for embedding queries.
    cache_size: int = 1024

@dataclass
class TrustConfig:
    """Trust policy defaults used before memories enter context."""

    minimum_confidence: float = 0.5

@dataclass
class MemoryAgentConfig:
    """Top-level configuration."""

    db_path: str = field(default_factory=lambda: str(default_memory_db_path()))

    memory_types: list[str] = field(
        default_factory=lambda: ["episodic", "semantic", "procedural", "preference"]
    )

    forcing: ForgettingConfig = field(default_factory=ForgettingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)

    @classmethod
    def default(cls) -> MemoryAgentConfig:
        return cls()
