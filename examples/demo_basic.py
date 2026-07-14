#!/usr/bin/env python
"""Basic demo: single-session memory agent interaction."""

import sys
import tempfile
from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine


def main():
    print("=" * 60)
    print("Alfredo MemoryAgent — Basic Demo")
    print("=" * 60)

    # Use temp db so we start clean.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    agent = None
    try:
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
        agent.init_session("basic-demo")

        interactions = [
            "Hi! My name is Manija",
            "I like programming in Python",
            "My favorite framework is Next.js",
            "I work as an Android developer",
            "What do you know about me?",
            "What is my favorite language?",
            "What framework do I like?",
            "Where do I work?",
        ]

        for user_input in interactions:
            print(f"\n{'─' * 50}")
            print(f"  You: {user_input}")

            result = agent.perceive(user_input)

            # Show extracted memories.
            if result["new_memories"]:
                print(f"  [{len(result['new_memories'])} new memories]")

            # Show recollections.
            if result["recollections"]:
                print(f"  Retrieved memories ({len(result['recollections'])}):")
                for r in result["recollections"][:3]:
                    print(f"    [{r.memory.memory_type}] {r.memory.content[:60]}")

            if result["archived"]:
                print(f"  [{result['archived']} archived memories]")

        # Final stats.
        print(f"\n{'=' * 60}")
        print("Final statistics:")
        stats = agent.get_stats()
        for key, val in stats.items():
            print(f"  {key}: {val}")
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
        run_cleanup(lambda: Path(db_path).unlink(missing_ok=True))
        if not primary_active and cleanup_error is not None:
            raise cleanup_error

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
