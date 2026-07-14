"""Read-only projection of active Alfredo memories into Markdown files.

The exporter deliberately depends only on :class:`MemoryStore`'s public read
methods.  SQLite remains the source of truth; files written here are a
one-way, replaceable projection for tools such as Obsidian.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


_FRONTMATTER_KEYS = (
    "memory_id",
    "memory_type",
    "confidence",
    "lifecycle",
    "namespace",
    "updated_at",
)


def _yaml_scalar(value: object) -> str:
    """Serialize one scalar without allowing YAML structure injection.

    Strings are emitted as JSON-quoted scalars.  JSON string escaping is also
    valid YAML and turns newlines, quotes, delimiter text, and control
    characters into data on one physical frontmatter line.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        # Non-finite values are not valid portable YAML numbers.  Quoting the
        # representation keeps the projection safe and deterministic.
        return json.dumps(repr(value), ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def _frontmatter(memory: MemoryRecord, namespace: str | None) -> str:
    """Build the fixed Alfredo frontmatter block for an active record."""
    values: dict[str, object] = {
        "memory_id": memory.id,
        "memory_type": memory.memory_type,
        "confidence": memory.confidence,
        "lifecycle": "active" if memory.is_active else "archived",
        "namespace": namespace,
        "updated_at": memory.updated_at,
    }
    return "\n".join(
        ["---"]
        + [f"{key}: {_yaml_scalar(values[key])}" for key in _FRONTMATTER_KEYS]
        + ["---"]
    )


def _selected_memories(
    store: MemoryStore,
    namespace: str | None,
    memory_ids: Iterable[int] | None,
) -> list[MemoryRecord]:
    """Read active records in exactly one namespace in stable ID order."""
    if memory_ids is None:
        records = store.get_all_active_memories(namespace=namespace)
    else:
        records = []
        seen: set[int] = set()
        for memory_id in memory_ids:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            record = store.get_memory(memory_id, namespace=namespace)
            if record is not None:
                records.append(record)

    # The public bulk read currently returns importance order.  Export order
    # must not depend on ranking or insertion details, so sort by filename ID.
    return sorted(
        (
            record
            for record in records
            if record.id is not None
            and record.is_active
            and record.namespace == namespace
        ),
        key=lambda record: record.id,
    )


def export_markdown(
    store: MemoryStore,
    output_dir: str | Path,
    namespace: str | None,
    memory_ids: Iterable[int] | None = None,
) -> None:
    """Export active same-namespace memories as deterministic ``<id>.md`` files.

    This function never writes to ``store``.  Stored content is copied as the
    Markdown body without normalization or mutation.  A body containing
    frontmatter-looking lines (including ``---``) is safe because it is placed
    after the exporter-owned closing delimiter; metadata is serialized as
    fixed, quoted scalar fields and cannot add frontmatter keys.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for memory in _selected_memories(store, namespace, memory_ids):
        document = f"{_frontmatter(memory, namespace)}\n\n{memory.content}"
        (destination / f"{memory.id}.md").write_text(document, encoding="utf-8")
