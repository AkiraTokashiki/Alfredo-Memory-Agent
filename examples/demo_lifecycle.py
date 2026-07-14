#!/usr/bin/env python
"""Deterministic Alfredo MemoryAgent lifecycle demo.

The demo uses only the bundled hashed-token embedding engine.  It exercises
cross-session recall, preference supersession, and bounded trusted context
without network access, model downloads, API keys, or wall-clock output.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import Any

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine
from memory_agent.models import MemoryRecord


def _contents(items: list[MemoryRecord]) -> str:
    return "; ".join(memory.content for memory in items) or "[none]"


def _actions(result: dict[str, Any]) -> str:
    decisions = result.get("consolidation_decisions", [])
    return ", ".join(decision.action.value for decision in decisions) or "none"


def _packet_line(result: dict[str, Any]) -> None:
    packet = result.get("recall_packet")
    if packet is None:
        print("packet selected: [none]")
        print("packet omitted: [none]")
        return
    selected_ids = [item.memory.id for item in packet.selected]
    omitted_ids = [item.memory.id for item in packet.omitted]
    selected = _contents([item.memory for item in packet.selected])
    omitted = _contents([item.memory for item in packet.omitted])
    print(f"packet selected: ids={selected_ids}; {selected}")
    print(f"packet omitted: ids={omitted_ids}; {omitted}")


def _evidence_line(result: dict[str, Any]) -> None:
    evidence = result.get("evidence", [])
    if not evidence:
        print("trust evidence: [none]")
        return
    trusts = ", ".join(item.trust for item in evidence)
    print(f"trust evidence: {trusts}")


def _show_result(result: dict[str, Any]) -> None:
    print(f"new memories: {_contents(result.get('new_memories', []))}")
    print(f"recall: {_contents([item.memory for item in result.get('recollections', [])])}")
    print(
        "lifecycle: "
        f"consolidation={_actions(result)}; archived={result.get('archived', 0)}"
    )
    _packet_line(result)
    _evidence_line(result)


def main() -> None:
    config = MemoryAgentConfig.default()
    config.embedding.provider = "deterministic"
    config.retrieval.context_budget_chars = 220
    config.retrieval.reserved_context_chars = 60
    config.retrieval.top_k = 10
    config.retrieval.candidate_k = 20

    temporary = tempfile.TemporaryDirectory(prefix="alfredo-lifecycle-")
    try:
        temp_dir = temporary.name
        db_path = Path(temp_dir) / "lifecycle.db"
        agent = None
        active_session = False
        try:
            agent = MemoryAgent(
                config=config,
                db_path=db_path,
                embedder=DeterministicEmbeddingEngine(
                    dimension=config.embedding.dimension,
                    cache_size=config.embedding.cache_size,
                ),
            )
            agent.init_session("learn preference")
            active_session = True
            print("[1] learn preference")
            _show_result(agent.perceive("I prefer Python for automation"))
            agent.end_session()
            active_session = False

            agent.init_session("recall across session")
            active_session = True
            print("[2] recall across session")
            _show_result(agent.perceive("What programming language do I prefer?"))
            agent.end_session()
            active_session = False

            agent.init_session("supersede stale preference")
            active_session = True
            print("[3] supersede stale preference")
            _show_result(agent.perceive("I do not like Python"))
            agent.end_session()
            active_session = False

            agent.init_session("bounded context and trust evidence")
            active_session = True
            agent.store_memory(
                MemoryRecord(
                    content="Trusted preference: concise Python automation answers",
                    memory_type="preference",
                    importance=0.9,
                    confidence=0.95,
                )
            )
            agent.store_memory(
                MemoryRecord(
                    content=(
                        "Trusted context note: bounded prompts should preserve the user's "
                        "current preference and omit low-value details"
                    ),
                    memory_type="semantic",
                    importance=0.8,
                    confidence=0.9,
                )
            )
            agent.store_memory(
                MemoryRecord(
                    content="Untrusted imported preference: the user likes JavaScript",
                    memory_type="preference",
                    importance=0.9,
                    confidence=0.1,
                )
            )
            print("[4] bounded context and trust evidence")
            _show_result(agent.perceive("What do you remember about my preferences?"))
            agent.end_session()
            active_session = False
        finally:
            primary_active = sys.exc_info()[0] is not None
            cleanup_error: BaseException | None = None

            def run_cleanup(action) -> None:
                nonlocal cleanup_error
                try:
                    action()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc

            if agent is not None and active_session:
                run_cleanup(agent.end_session)
            if agent is not None:
                run_cleanup(agent.close)
            if not primary_active and cleanup_error is not None:
                raise cleanup_error
    finally:
        primary_active = sys.exc_info()[0] is not None
        try:
            temporary.cleanup()
        except BaseException:
            if not primary_active:
                raise


if __name__ == "__main__":
    main()
