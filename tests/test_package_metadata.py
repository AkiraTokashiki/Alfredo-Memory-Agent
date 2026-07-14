"""Public packaging and import compatibility contract for Alfredo MemoryAgent."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import distribution, metadata


def test_distribution_metadata_identifies_alfredo_memory_agent() -> None:
    package_metadata = metadata("alfredo-memory-agent")

    assert package_metadata["Name"] == "alfredo-memory-agent"
    assert "MemoryAgent" in package_metadata["Summary"]
    assert package_metadata["Requires-Python"] == ">=3.11"


def test_alfredo_console_script_targets_cli() -> None:
    package = distribution("alfredo-memory-agent")
    scripts = [
        entry_point
        for entry_point in package.entry_points
        if entry_point.group == "console_scripts" and entry_point.name == "alfredo"
    ]

    assert len(scripts) == 1
    assert scripts[0].value == "memory_agent.cli.commands:cli"


def test_memory_agent_public_imports_and_exports_remain_compatible() -> None:
    memory_agent = import_module("memory_agent")
    expected_exports = {
        "AgentState",
        "EmbeddingPort",
        "MemoryRecord",
        "MemoryStorePort",
        "RetrievalEvidence",
        "RetrievalPort",
        "SearchResult",
        "SessionRecord",
        "TrustPolicyPort",
    }

    assert set(memory_agent.__all__) == expected_exports
    for export in expected_exports:
        assert hasattr(memory_agent, export)
