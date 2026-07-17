"""Optional dependency detection without importing heavyweight modules."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    available: bool
    detail: str
    install_command: str | None = None


def detect_capabilities() -> tuple[CapabilityStatus, ...]:
    mcp = importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("httpx") is not None
    semantic = importlib.util.find_spec("sentence_transformers") is not None
    return (
        CapabilityStatus("core", True, "Core Alfredo functionality is available."),
        CapabilityStatus("mcp", mcp, "MCP dependencies are installed." if mcp else "Install the MCP extra.", "python -m pip install -e \".[mcp]\""),
        CapabilityStatus("semantic", semantic, "Semantic embedding support is available." if semantic else "Install the semantic extra.", "python -m pip install -e \".[semantic]\""),
    )


def missing_capability_text(status: CapabilityStatus) -> str:
    return f"{status.detail} Run: {status.install_command}" if status.install_command else status.detail
