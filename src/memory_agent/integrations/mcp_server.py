"""MCP Server for MemoryAgent.

Exposes MemoryAgent as MCP tools so any MCP client (Hermes, Claude Desktop, Cursor)
can use its persistent memory capabilities.

Usage:
    python -m memory_agent mcp                  # stdio transport (for Hermes config)
    python -m memory_agent mcp --http --port 8080      # HTTP transport (for remote clients)

Hermes config to add:
    mcp_servers:
      memory-agent:
        command: python
        args: ["-m", "memory_agent", "mcp"]

DB path: set MEMORY_AGENT_DB env var, or defaults to ./memory_agent.db
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.models import MemoryRecord

# ---------------------------------------------------------------------------
# Singleton agent with thread safety
# ---------------------------------------------------------------------------

_agent: MemoryAgent | None = None
_agent_lock = threading.Lock()


def _resolve_db_path() -> Path:
    """Resolve DB path: env var MEMORY_AGENT_DB, or cwd/memory_agent.db."""
    env_path = os.environ.get("MEMORY_AGENT_DB")
    if env_path:
        return Path(env_path).resolve()
    return Path.cwd() / "memory_agent.db"


def _get_agent() -> MemoryAgent:
    """Lazy init with thread lock so the model is only loaded once."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:  # double-check after acquiring lock
                config = MemoryAgentConfig.default()
                db_path = _resolve_db_path()
                _agent = MemoryAgent(config=config, db_path=db_path)
    return _agent


def _ensure_session(*, namespace: str | None = None) -> None:
    """Auto-start a session if none active, preserving namespace isolation."""
    agent = _get_agent()
    if agent.state.session_id is None:
        agent.init_session("mcp-auto", namespace=namespace)
def _public(value: Any) -> Any:
    """Convert public model values into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _public(value.to_dict())
    if is_dataclass(value):
        return _public(asdict(value))
    if isinstance(value, dict):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _result_public(result: Any) -> dict[str, Any]:
    """Keep the existing flat MCP result fields while using model ``to_dict``."""
    if isinstance(result, dict):
        payload = _public(result)
        memory = payload.get("memory")
        if isinstance(memory, dict):
            flattened = {
                "id": memory.get("id"),
                "content": memory.get("content", ""),
                "type": memory.get("memory_type", memory.get("type")),
                "importance": memory.get("importance"),
                "strength": memory.get("strength"),
                "access_count": memory.get("access_count", 0),
                "tags": memory.get("tags", []),
                "score": payload.get("score", 0.0),
                "scores": {
                    "semantic": payload.get("semantic_score", 0.0),
                    "recency": payload.get("recency_score", 0.0),
                    "importance": payload.get("importance_score", 0.0),
                    "strength": payload.get("strength_score", 0.0),
                },
            }
            if isinstance(payload.get("evidence"), dict):
                flattened["evidence"] = payload["evidence"]
                flattened["trust"] = payload["evidence"].get("trust", "unknown")
                flattened["reason"] = payload["evidence"].get("reason", "")
            return flattened
        return payload
    memory = result.memory
    payload = {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type,
        "importance": memory.importance,
        "strength": memory.strength,
        "access_count": memory.access_count,
        "tags": list(memory.tags),
        "score": result.score,
        "scores": {
            "semantic": result.semantic_score,
            "recency": result.recency_score,
            "importance": result.importance_score,
            "strength": result.strength_score,
        },
    }
    if result.evidence is not None:
        payload["evidence"] = result.evidence.to_dict()
        payload["trust"] = result.trust
        payload["reason"] = result.reason
    return _public(payload)


def _evidence_public(result: Any, reason: str | None = None) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = _public(result)
        if reason:
            payload["reason"] = reason
        return payload
    payload = {
        "id": result.memory.id,
        **(result.evidence.to_dict() if result.evidence is not None else {}),
    }
    if reason:
        payload["reason"] = reason
    return _public(payload)


# Create MCP server
mcp = FastMCP(
    "MemoryAgent",
    instructions="""MemoryAgent — persistent memory system for AI agents.

Provides tools to store and retrieve memories across sessions with:
- Semantic search (sentence-transformers embeddings)
- Ebbinghaus forgetting curve (memories decay over time unless reinforced)
- Multi-type memory (episodic, semantic, preferences, procedures)
- MMR diversity to avoid near-duplicate results
""",
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="memory__perceive",
    description="Process user input: extract, store, and retrieve memories in one step. "
    "Returns recollections relevant to the input plus any new memories extracted.",
)
async def memory_perceive(
    user_input: str,
    top_k: int = 5,
    namespace: str | None = None,
) -> str:
    """Process input through MemoryAgent with an explicit namespace."""
    _ensure_session(namespace=namespace)
    agent = _get_agent()
    result = agent.perceive(user_input, namespace=namespace)
    packet = result.get("recall_packet")
    recollections = result.get("recollections", [])[:top_k]
    selected_ids = (
        packet.selected_ids
        if packet is not None
        else result.get(
            "selected_ids",
            [item.get("id") for item in recollections if isinstance(item, dict)],
        )
    )
    dropped_ids = (
        packet.dropped_ids
        if packet is not None
        else result.get("dropped_ids", [])
    )
    if packet is not None:
        evidence = [
            _evidence_public(item, packet.reasons.get(id(item)))
            for item in packet.selected + packet.omitted
        ]
    else:
        evidence = _public(result.get("evidence", []))
    effective_namespace = (
        result.get("lifecycle", {}).get("namespace", namespace)
        if isinstance(result.get("lifecycle"), dict)
        else namespace
    )
    output = {
        "namespace": effective_namespace,
        "recollections": [_result_public(item) for item in recollections],
        "new_memories": [
            {
                "id": item.id,
                "content": item.content[:80],
                "type": item.memory_type,
                "namespace": item.namespace,
            }
            if hasattr(item, "id")
            else _public(item)
            for item in result.get("new_memories", [])
        ],
        "selected_ids": selected_ids,
        "dropped_ids": dropped_ids,
        "evidence": evidence,
        "stats": _public(
            {
                "turn": result.get("turn_count", 0),
                "total_memories": result.get("total_memories", 0),
                "archived": result.get("archived", 0),
            }
        ),
        "lifecycle": _public(
            result.get(
                "lifecycle",
                {"operation": "perceive", "status": "processed", "namespace": namespace},
            )
        ),
    }
    return json.dumps(_public(output), ensure_ascii=False, indent=2, allow_nan=False)


@mcp.tool(
    name="memory__search",
    description="Semantic search across all stored memories. "
    "Finds relevant memories even without exact keyword matches.",
)
async def memory_search(
    query: str,
    top_k: int = 5,
    memory_type: str | None = None,
    namespace: str | None = None,
) -> str:
    """Search memories through the trust-aware MemoryAgent facade."""
    _ensure_session(namespace=namespace)
    agent = _get_agent()
    payload = agent.search_memories(
        query,
        top_k=top_k,
        memory_type=memory_type,
        namespace=namespace,
    )

    payload = _public(payload)
    payload["results"] = [
        _result_public(item) for item in payload.get("results", [])
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


@mcp.tool(
    name="memory__store",
    description="Manually store a memory. Use when you want to explicitly "
    "save a fact, preference, or experience for future recall.",
)
async def memory_store(
    content: str,
    memory_type: str = "semantic",
    importance: float = 0.5,
    tags: str = "",
    namespace: str | None = None,
) -> str:
    """Store a memory through the namespace-aware MemoryAgent facade."""
    _ensure_session(namespace=namespace)
    agent = _get_agent()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    importance = max(0.0, min(1.0, importance))
    mem = MemoryRecord(
        content=content,
        memory_type=memory_type,
        importance=importance,
        tags=tag_list,
        namespace=namespace,
    )
    mid = agent.store_memory(mem, namespace=namespace)
    effective_namespace = (
        mem.namespace
        if mem.namespace is not None
        else getattr(agent, "namespace", namespace)
    )
    return json.dumps(
        {
            "id": mid,
            "content": content[:60],
            "type": memory_type,
            "importance": importance,
            "tags": tag_list,
            "namespace": effective_namespace,
            "status": "stored",
            "lifecycle": {
                "operation": "store",
                "status": "stored",
                "namespace": effective_namespace,
            },
        },
        ensure_ascii=False,
        allow_nan=False,
    )


@mcp.tool(
    name="memory__stats",
    description="Get memory agent statistics: total memories, type distribution, "
    "decay lifespans, embedding count, and archival info.",
)
async def memory_stats(namespace: str | None = None) -> str:
    """Return namespace-scoped memory statistics."""
    _ensure_session(namespace=namespace)
    agent = _get_agent()
    stats = agent.get_stats(namespace=namespace)
    return json.dumps(_public(stats), ensure_ascii=False, indent=2, allow_nan=False)


@mcp.tool(
    name="memory__forget",
    description="Delete a specific memory by ID. Use when a memory is wrong or obsolete.",
)
async def memory_forget(memory_id: int, namespace: str | None = None) -> str:
    """Archive a memory through the MemoryAgent facade."""
    agent = _get_agent()
    result = agent.forget_memory(memory_id, namespace=namespace)
    return json.dumps(_public(result), ensure_ascii=False, allow_nan=False)


@mcp.tool(
    name="memory__reinforce",
    description="Reinforce a memory by ID — boosts its recall strength. "
    "Use when the user confirms a memory is still relevant.",
)
async def memory_reinforce(memory_id: int, namespace: str | None = None) -> str:
    """Reinforce a memory through the MemoryAgent facade."""
    agent = _get_agent()
    result = agent.reinforce_memory(memory_id, namespace=namespace)
    return json.dumps(_public(result), ensure_ascii=False, allow_nan=False)




# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource(
    uri="memory://recent",
    name="Recent Memories",
    description="The 10 most recent active memories.",
    mime_type="application/json",
)
async def recent_memories() -> str:
    """Return the 10 most recent active memories."""
    agent = _get_agent()
    memories = agent.list_memories()[:10]
    output = [
        {
            "id": m.id,
            "content": m.content[:80],
            "type": m.memory_type,
            "importance": m.importance,
            "strength": round(m.strength, 3),
            "created_at": m.created_at,
            "namespace": m.namespace,
        }
        for m in memories
    ]
    return json.dumps(_public(output), ensure_ascii=False, indent=2, allow_nan=False)

@mcp.resource(
    uri="memory://stats",
    name="Memory Stats",
    description="Current memory agent statistics.",
    mime_type="application/json",
)
async def stats_resource() -> str:
    """Return memory stats as a resource."""
    agent = _get_agent()
    stats = agent.get_stats()
    return json.dumps(_public(stats), ensure_ascii=False, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="memory-assisted",
    description="Load relevant memories before responding to a user query.",
)
async def memory_assisted_prompt(
    query: str, namespace: str | None = None
) -> str:
    """Generate a prompt using trust-filtered namespace-scoped memories."""
    agent = _get_agent()
    payload = agent.search_memories(query, top_k=5, namespace=namespace)
    results = payload.get("results", [])
    if not results:
        return f"## Query\n\n{query}\n\n*(No relevant memories found)*"
    memories_text = "\n".join(
        f"- [{item.get('memory', {}).get('memory_type', item.get('type', 'memory'))}] "
        f"{item.get('memory', {}).get('content', item.get('content', ''))}"
        for item in results
    )
    return f"""## Relevant Memories
{memories_text}

## Query
{query}

## Instruction
Using the relevant memories above to inform your response, answer the user's query.
If memories suggest a preference or past experience, reference it naturally."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_mcp_server(host: str | None = None, port: int | None = None) -> None:
    """Start the MCP server.

    Args:
        host: HTTP host (HTTP mode). None = stdio mode.
        port: HTTP port (HTTP mode). None = stdio mode.
    """
    # Pre-warm agent so first MCP call isn't slow
    _get_agent()

    if host and port:
        print(f"MemoryAgent MCP Server at http://{host}:{port}/mcp", file=sys.stderr)
        previous_host = mcp.settings.host
        previous_port = mcp.settings.port
        mcp.settings.host = host
        mcp.settings.port = port
        try:
            mcp.run(transport="streamable-http")
        finally:
            mcp.settings.host = previous_host
            mcp.settings.port = previous_port
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
