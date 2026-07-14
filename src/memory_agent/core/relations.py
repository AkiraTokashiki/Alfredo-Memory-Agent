"""Typed memory relation validation helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from memory_agent.models import MemoryRelation


ALLOWED_RELATION_TYPES = frozenset(
    {"related_to", "supports", "supersedes", "contradicts", "derived_from"}
)


class RelationManager:
    """Validate relation records and endpoint namespace invariants.

    SQLite persistence remains the responsibility of :class:`MemoryStore`; this
    class contains the domain checks so callers cannot accidentally bypass them.
    """

    allowed_types = ALLOWED_RELATION_TYPES

    @classmethod
    def validate_type(cls, relation_type: str) -> None:
        if relation_type not in cls.allowed_types:
            raise ValueError(f"unknown relation type: {relation_type!r}")

    @classmethod
    def validate_confidence(cls, confidence: float) -> None:
        try:
            finite = math.isfinite(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("relation confidence must be finite") from exc
        if not finite:
            raise ValueError("relation confidence must be finite")

    @classmethod
    def validate_relation(cls, relation: MemoryRelation) -> None:
        if not isinstance(relation, MemoryRelation):
            raise TypeError("relation must be a MemoryRelation")
        if relation.source_id is None or relation.target_id is None:
            raise ValueError("relation endpoints are required")
        if relation.source_id == relation.target_id:
            raise ValueError("memory relations cannot link a memory to itself")
        cls.validate_type(relation.relation_type)
        cls.validate_confidence(relation.confidence)

    @classmethod
    def validate_endpoints(
        cls,
        relation: MemoryRelation,
        source: Mapping[str, Any] | None,
        target: Mapping[str, Any] | None,
    ) -> None:
        cls.validate_relation(relation)
        if source is None or target is None:
            raise ValueError("relation endpoints must reference existing memories")
        source_namespace = source["namespace"]
        target_namespace = target["namespace"]
        if source_namespace != target_namespace:
            raise ValueError("relation endpoints must share a namespace")
        if relation.namespace != source_namespace:
            raise ValueError("relation namespace does not match memory endpoints")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryRelation:
        """Construct and validate a relation from a mapping."""
        relation = MemoryRelation.from_dict(dict(payload))
        cls.validate_relation(relation)
        return relation
