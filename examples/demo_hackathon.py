"""Hackathon demo: persistent memory, stale preference replacement, and context budget."""

from __future__ import annotations

from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.models import MemoryRecord


DB_PATH = Path("hackathon_demo.db")


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
    DB_PATH.unlink(missing_ok=True)
    agent = MemoryAgent(db_path=DB_PATH)

    agent.init_session("session 1")
    print_turn(
        "Session 1: learn preferences",
        agent.perceive("Me gusta Python y prefiero respuestas concisas"),
    )
    agent.end_session()

    agent.init_session("session 2")
    print_turn("Session 2: recall preference", agent.perceive("Que lenguaje me gusta?"))
    agent.end_session()

    agent.init_session("session 3")
    print_turn("Session 3: update stale preference", agent.perceive("No me gusta Python"))
    agent.end_session()

    agent.init_session("session 4")
    for idx in range(20):
        agent.store_memory(
            MemoryRecord(content=f"ruido de baja importancia {idx}", importance=0.1)
        )
    print_turn(
        "Session 4: bounded recall after noise",
        agent.perceive("Que recuerdas de mis preferencias?"),
    )

    stats = agent.get_stats()
    print("\n=== Stats ===")
    print(f"active: {stats['total_active']}")
    print(f"archived: {stats['archived']}")
    print(f"types: {stats['type_distribution']}")

    agent.close()


if __name__ == "__main__":
    main()
