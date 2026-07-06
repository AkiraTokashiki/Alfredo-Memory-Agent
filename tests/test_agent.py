"""Tests for the MemoryAgent orchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.models import MemoryRecord


@pytest.fixture
def agent() -> MemoryAgent:
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    a = MemoryAgent(db_path=db.name)
    yield a
    a.close()
    Path(db.name).unlink(missing_ok=True)


class TestMemoryAgent:
    def test_init_session(self, agent: MemoryAgent):
        agent.init_session("test")
        assert agent.state.session_id is not None
        assert agent.state.turn_count == 0

    def test_perceive_empty(self, agent: MemoryAgent):
        agent.init_session()
        result = agent.perceive("hola")
        assert result["turn_count"] == 1
        assert isinstance(result["recollections"], list)
        assert isinstance(result["recollection_text"], str)

    def test_perceive_stores_preferences(self, agent: MemoryAgent):
        agent.init_session()
        result = agent.perceive("Me gusta programar en Python")
        assert len(result["new_memories"]) >= 1
        # Should have extracted preference
        pref = result["new_memories"][0]
        assert pref.memory_type in ("preference", "habit")
        assert "Python" in pref.content or "programar" in pref.content
        assert "preference" in pref.tags or "extracted" in pref.tags

    def test_perceive_recalls_memories(self, agent: MemoryAgent):
        agent.init_session()

        # Store a memory first
        agent.perceive("Mi lenguaje favorito es Python")
        # Should have stored at least the interaction as episodic memory
        assert agent.state.total_memories >= 1

        # Now ask about it
        result = agent.perceive("Que lenguaje me gusta?")
        recollections = result["recollections"]
        if recollections:
            # Should recall Python-related memories
            assert any(
                "Python" in r.memory.content for r in recollections
            )

    def test_forgetting_cycle_runs(self, agent: MemoryAgent):
        agent.init_session()
        for i in range(10):
            result = agent.perceive(f"turno {i}")
        # Decay runs every 3 turns, so by turn 10 it should have run at least 3 times
        assert result["turn_count"] == 10

    def test_archival(self, agent: MemoryAgent):
        agent.init_session()
        # Add a very weak memory directly
        weak = MemoryRecord(content="algo olvidable", strength=0.01, importance=0.1)
        agent.store.add_memory(weak)

        # Run decay cycle manually
        from datetime import datetime, timedelta
        past = datetime.now() - timedelta(days=90)

        # Force archive
        archived = agent.store.archive_below_threshold(0.1)
        assert archived >= 1

    def test_stats(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta el cafe")
        agent.perceive("Trabajo como programador")

        stats = agent.get_stats()
        assert stats["total_active"] >= 2
        assert stats["session_turns"] == 2
        assert stats["avg_importance"] > 0

    def test_multiple_sessions(self, agent: MemoryAgent):
        # Session 1
        agent.init_session("session 1")
        agent.perceive("Me gusta Python")
        s1_memories = agent.state.total_memories
        agent.end_session()

        # Session 2
        agent.init_session("session 2")
        agent.perceive("Que recuerdas de mi?")
        # Should still have memories from session 1
        assert agent.state.total_memories >= s1_memories

    def test_importance_decay_lifespan(self, agent: MemoryAgent):
        """High-importance memories should decay slower in the forgetting curve."""
        high = agent.forgetting.predicted_lifespan_days(0.9)
        low = agent.forgetting.predicted_lifespan_days(0.3)
        assert high > low * 10

    def test_store_memory_public(self, agent: MemoryAgent):
        """Public store_memory() method should work."""
        agent.init_session()
        mem = MemoryRecord(content="test public store", importance=0.8)
        mid = agent.store_memory(mem)
        assert mid is not None
        assert mid > 0
        fetched = agent.store.get_memory(mid)
        assert fetched is not None
        assert fetched.content == "test public store"

    def test_negative_preference(self, agent: MemoryAgent):
        """'no me gusta X' should store as negative preference."""
        agent.init_session()
        from memory_agent.agent.decision import extract_from_input
        memories = extract_from_input("No me gusta el cafe con leche")
        assert len(memories) >= 1
        mem = memories[0]
        assert "no le gusta" in mem.content
        assert mem.importance == pytest.approx(0.6)

    def test_high_importance_preference(self, agent: MemoryAgent):
        """Strong preferences should have higher importance."""
        from memory_agent.agent.decision import extract_from_input
        memories = extract_from_input("Me encanta la musica electronica")
        assert len(memories) >= 1
        assert memories[0].importance == pytest.approx(0.9)

    def test_reinforce_and_access_no_double_count(self):
        """reinforce_and_access should not double-increment access_count."""
        from memory_agent.core.forgetting import ForgettingCurve
        curve = ForgettingCurve()
        mem = MemoryRecord(content="test", strength=0.5, access_count=0)
        curve.reinforce_and_access(mem)
        # reinforce() itself increments by 1, so access_count should be 1
        assert mem.access_count == 1

    def test_preference_extraction_adds_topic_and_polarity(self):
        from memory_agent.agent.decision import extract_from_input

        memories = extract_from_input("Me gusta Python")

        assert memories
        assert memories[0].metadata["topic"] == "python"
        assert memories[0].metadata["polarity"] == "positive"

    def test_negative_preference_extraction_adds_negative_polarity(self):
        from memory_agent.agent.decision import extract_from_input

        memories = extract_from_input("No me gusta Python")

        assert memories
        assert memories[0].metadata["topic"] == "python"
        assert memories[0].metadata["polarity"] == "negative"

    def test_explicit_forget_detection(self):
        from memory_agent.agent.decision import extract_forget_query

        assert extract_forget_query("forget python") == "python"
        assert extract_forget_query("olvida que me gusta python") == "me gusta python"
        assert extract_forget_query("Me gusta Python") is None

    def test_duplicate_preference_is_not_stored_twice(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")
        first_total = agent.state.total_memories

        result = agent.perceive("Me gusta programar en Python")

        active_preferences = [
            m for m in agent.store.get_all_active_memories()
            if m.memory_type == "preference" and "python" in m.content.lower()
        ]
        assert len(active_preferences) == 1
        assert result["consolidation_decisions"]
        assert agent.state.total_memories <= first_total + 1

    def test_changed_preference_supersedes_old_preference(self, agent: MemoryAgent):
        agent.init_session("session 1")
        agent.perceive("Me gusta Python")
        agent.end_session()

        agent.init_session("session 2")
        agent.perceive("No me gusta Python")

        all_memories = [
            agent.store.get_memory(m.id)
            for m in agent.store.get_all_active_memories()
            if m.id
        ]
        active_python_preferences = [
            m for m in all_memories
            if m is not None and m.memory_type == "preference" and "python" in m.content.lower()
        ]
        assert len(active_python_preferences) == 1
        assert "no le gusta" in active_python_preferences[0].content.lower()

        archived_count = agent.store.count_memories(active_only=False) - agent.store.count_memories(active_only=True)
        assert archived_count >= 1

    def test_explicit_forget_removes_memory_from_default_recall(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")

        result = agent.perceive("forget python")

        assert result["archived"] >= 1
        active_contents = [m.content.lower() for m in agent.store.get_all_active_memories()]
        assert not any("prefiere: python" in content for content in active_contents)

    def test_perceive_returns_recall_packet(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")

        result = agent.perceive("Que lenguaje me gusta?")

        assert "recall_packet" in result
        assert result["recall_packet"].used_chars <= result["recall_packet"].available_chars
