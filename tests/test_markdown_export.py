"""Black-box contracts for the Markdown/Obsidian memory export."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord
from memory_agent.integrations.markdown_export import export_markdown


@pytest.fixture
def store(tmp_path: Path):
    """Provide a real, isolated SQLite-backed store for each export test."""
    memory_store = MemoryStore(tmp_path / "memory.db")
    memory_store.initialize()
    yield memory_store
    memory_store.close()


def _frontmatter_and_body(document: str) -> tuple[dict[str, object], str]:
    """Read the small YAML frontmatter subset emitted by the exporter."""
    assert document.startswith("---\n")
    closing = document.find("\n---\n", 4)
    assert closing != -1, "export must close frontmatter before the memory body"

    values: dict[str, object] = {}
    for line in document[4:closing].splitlines():
        key, separator, raw_value = line.partition(":")
        assert separator, f"invalid frontmatter line: {line!r}"
        raw_value = raw_value.strip()
        if raw_value in {"null", "~"}:
            value: object = None
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                value = raw_value.strip("'\"")
        values[key] = value
    return values, document[closing + len("\n---\n") :]


def _add(
    store: MemoryStore,
    content: str,
    *,
    namespace: str = "tenant-a",
    memory_type: str = "preference",
    confidence: float | None = 0.9,
    importance: float = 0.5,
    is_active: bool = True,
) -> int:
    return store.add_memory(
        MemoryRecord(
            content=content,
            namespace=namespace,
            memory_type=memory_type,
            confidence=confidence,
            importance=importance,
            is_active=is_active,
        )
    )


def test_export_includes_only_active_records_in_namespace_with_frontmatter(
    store: MemoryStore, tmp_path: Path
) -> None:
    """The all-record export is namespace-scoped and carries stable Alfredo metadata."""
    selected_id = _add(
        store,
        "The user prefers Python.",
        memory_type="preference",
        confidence=0.91,
        importance=0.9,
    )
    second_id = _add(
        store,
        "Use concise answers.",
        memory_type="procedural",
        confidence=0.72,
        importance=0.8,
    )
    archived_id = _add(store, "Old preference", importance=1.0)
    store.archive_memory(archived_id, namespace="tenant-a", reason="superseded")
    cross_namespace_id = _add(
        store, "Other tenant secret", namespace="tenant-b", importance=1.0
    )

    output_dir = tmp_path / "vault"
    export_markdown(store, output_dir, "tenant-a")

    assert sorted(path.name for path in output_dir.iterdir()) == [
        f"{selected_id}.md",
        f"{second_id}.md",
    ]
    assert f"{archived_id}.md" not in {path.name for path in output_dir.iterdir()}
    assert f"{cross_namespace_id}.md" not in {
        path.name for path in output_dir.iterdir()
    }

    selected_record = store.get_memory(selected_id, namespace="tenant-a")
    assert selected_record is not None
    frontmatter, body = _frontmatter_and_body(
        (output_dir / f"{selected_id}.md").read_text(encoding="utf-8")
    )
    assert set(frontmatter) == {
        "memory_id",
        "memory_type",
        "confidence",
        "lifecycle",
        "namespace",
        "updated_at",
    }
    assert frontmatter["memory_id"] == selected_id
    assert frontmatter["memory_type"] == "preference"
    assert frontmatter["confidence"] == pytest.approx(0.91)
    assert frontmatter["lifecycle"] == "active"
    assert frontmatter["namespace"] == "tenant-a"
    assert frontmatter["updated_at"] == selected_record.updated_at
    assert "The user prefers Python." in body


def test_export_safely_separates_frontmatter_from_markdown_content(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Frontmatter-looking content cannot inject or overwrite export metadata."""
    content = (
        "# A heading\n"
        "---\n"
        "memory_id: 999\n"
        "namespace: tenant-b\n"
        "[untrusted](javascript:alert(1))"
    )
    memory_id = _add(store, content, confidence=0.64)

    output_dir = tmp_path / "vault"
    export_markdown(store, output_dir, "tenant-a")

    frontmatter, body = _frontmatter_and_body(
        (output_dir / f"{memory_id}.md").read_text(encoding="utf-8")
    )
    assert frontmatter["memory_id"] == memory_id
    assert frontmatter["namespace"] == "tenant-a"
    assert "memory_id: 999" not in "\n".join(
        f"{key}: {value}" for key, value in frontmatter.items()
    )
    assert content in body


def test_export_is_read_only_and_repeated_bytes_are_identical(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Exporting never mutates SQLite and the same inputs produce byte-identical files."""
    first_id = _add(store, "First", confidence=0.8, importance=0.4)
    second_id = _add(store, "Second", confidence=0.7, importance=0.6)
    before_rows = [
        tuple(row)
        for row in store.conn.execute(
            "SELECT id, content, is_active, namespace, confidence, updated_at "
            "FROM memories ORDER BY id"
        )
    ]
    before_changes = store.conn.total_changes

    output_dir = tmp_path / "vault"
    export_markdown(store, output_dir, "tenant-a")
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.glob("*.md"))
    }
    export_markdown(store, output_dir, "tenant-a")
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.glob("*.md"))
    }

    after_rows = [
        tuple(row)
        for row in store.conn.execute(
            "SELECT id, content, is_active, namespace, confidence, updated_at "
            "FROM memories ORDER BY id"
        )
    ]
    assert first_bytes == second_bytes
    assert set(first_bytes) == {f"{first_id}.md", f"{second_id}.md"}
    assert after_rows == before_rows
    assert store.conn.total_changes == before_changes


def test_export_memory_ids_filters_within_namespace_and_active_lifecycle(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Explicit IDs select only active records owned by the requested namespace."""
    wanted_id = _add(store, "Wanted", confidence=0.83)
    other_id = _add(store, "Not selected", confidence=0.82)
    archived_id = _add(store, "Archived", confidence=0.81)
    store.archive_memory(archived_id, namespace="tenant-a", reason="expired")
    cross_namespace_id = _add(
        store, "Cross namespace", namespace="tenant-b", confidence=0.99
    )

    output_dir = tmp_path / "vault"
    export_markdown(
        store,
        output_dir,
        "tenant-a",
        memory_ids=[wanted_id, other_id, archived_id, cross_namespace_id, 999999],
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [f"{wanted_id}.md"]
    assert "Wanted" in (output_dir / f"{wanted_id}.md").read_text(encoding="utf-8")
