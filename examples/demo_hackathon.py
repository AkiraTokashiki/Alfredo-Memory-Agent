"""Hackathon demo: persistent memory, stale preference replacement, and context budget."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine
from memory_agent.models import MemoryRecord




def print_turn(title: str, result: dict) -> None:
    print(f"\n=== {title} ===")
    print(result["recollection_text"] or "[no recollections]")
    print(f"active memories: {result['total_memories']}")
    print(f"archived this turn: {result['archived']}")
    packet = result.get("recall_packet")
    if packet is not None:
        print(f"context budget: {packet.used_chars}/{packet.available_chars} chars")
        print(f"omitted memories: {len(packet.omitted)}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="alfredo-hackathon-") as temp_dir:
        db_path = Path(temp_dir) / "hackathon_demo.db"
        agent = None
        try:
            print("Alfredo MemoryAgent — Hackathon Demo")
            config = MemoryAgentConfig.default()
            config.embedding.provider = "deterministic"
            agent = MemoryAgent(
                config=config,
                db_path=db_path,
                embedder=DeterministicEmbeddingEngine(
                    dimension=config.embedding.dimension,
                    cache_size=config.embedding.cache_size,
                ),
            )

            agent.init_session("session 1")
            print_turn(
                "Session 1: learn preferences",
                agent.perceive("I like Python and I prefer concise answers"),
            )
            agent.end_session()

            agent.init_session("session 2")
            print_turn("Session 2: recall preference", agent.perceive("What language do I like?"))
            agent.end_session()

            agent.init_session("session 3")
            print_turn("Session 3: update stale preference", agent.perceive("I do not like Python"))
            agent.end_session()

            agent.init_session("session 4")
            for idx in range(20):
                agent.store_memory(
                    MemoryRecord(content=f"low-importance noise {idx}", importance=0.1)
                )
            print_turn(
                "Session 4: bounded recall after noise",
                agent.perceive("What do you remember about my preferences?"),
            )

            stats = agent.get_stats()
            print("\n=== Stats ===")
            print(f"active: {stats['total_active']}")
            print(f"archived: {stats['archived']}")
            print(f"types: {stats['type_distribution']}")
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

            if agent is not None:
                run_cleanup(agent.end_session)
                run_cleanup(agent.close)
            run_cleanup(lambda: db_path.unlink(missing_ok=True))
            if not primary_active and cleanup_error is not None:
                raise cleanup_error


if __name__ == "__main__":
    main()
