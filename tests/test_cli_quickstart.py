"""Offline first-run CLI tests."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

import pytest

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.cli import commands as commands_module
from memory_agent.cli.commands import cli
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.deterministic_embeddings import DeterministicEmbeddingEngine
from memory_agent.models import MemoryRecord


def test_configured_offline_provider_selects_engine_without_cli() -> None:
    with TemporaryDirectory(prefix="memory-agent-test-") as temp_dir:
        config = MemoryAgentConfig.default()
        config.db_path = str(Path(temp_dir) / "memory.db")
        config.embedding.provider = "deterministic"

        agent = MemoryAgent(config=config)

        assert isinstance(agent.embeddings, DeterministicEmbeddingEngine)
        agent.store.close()



def test_explicit_provider_encode_failure_is_not_silenced() -> None:
    class FailingProvider:
        model_name = "production-model"
        provider = "sentence-transformers"

        def encode(self, text: str) -> bytes:
            raise RuntimeError("model encode failed")

    with TemporaryDirectory(prefix="memory-agent-test-") as temp_dir:
        config = MemoryAgentConfig.default()
        config.db_path = str(Path(temp_dir) / "memory.db")
        agent = MemoryAgent(config=config, embedder=FailingProvider())

        with pytest.raises(RuntimeError, match="provider failed"):
            agent.store_memory(MemoryRecord(content="must index"))
        agent.store.close()

def test_offline_quickstart_stores_and_recalls_without_model_or_api_key(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    runner = CliRunner()

    result = runner.invoke(cli, ["--offline", "quickstart"])

    assert result.exit_code == 0, result.output
    assert "offline" in result.output.lower()
    assert "python" in result.output.lower()
    assert "remember" in result.output.lower()


def test_quickstart_fails_when_sqlite_recall_is_empty(monkeypatch) -> None:
    class FakeStore:
        def close(self) -> None:
            pass

    class EmptyRecallAgent:
        store = FakeStore()

        def init_session(self, label: str) -> None:
            pass

        def end_session(self) -> None:
            pass

        def perceive(self, text: str) -> dict[str, str]:
            return {"recollection_text": ""}

    monkeypatch.setattr(commands_module, "_get_agent", lambda ctx: EmptyRecallAgent())
    result = CliRunner().invoke(cli, ["--offline", "quickstart"])

    assert result.exit_code != 0
    assert "could not recall" in result.output.lower()
    assert "i prefer python for automation" not in result.output.lower()


def test_quickstart_closes_store_when_session_end_fails(monkeypatch) -> None:
    closed = []

    class FailingStore:
        def close(self) -> None:
            closed.append(True)
            raise RuntimeError("store close failed")

    class FailingAgent:
        store = FailingStore()

        def init_session(self, label: str) -> None:
            pass

        def end_session(self) -> None:
            raise RuntimeError("session close failed")

        def perceive(self, text: str) -> dict[str, str]:
            return {"recollection_text": "[Retrieved memories]: remembered"}

    monkeypatch.setattr(commands_module, "_get_agent", lambda ctx: FailingAgent())
    result = CliRunner().invoke(cli, ["--offline", "quickstart"])

    assert result.exit_code != 0
    assert result.exception is not None
    assert "session close failed" in str(result.exception)
    assert "store close failed" not in str(result.exception)
    assert closed == [True]


def test_quickstart_attempts_all_cleanup_for_base_exceptions(monkeypatch) -> None:
    events = []

    class BaseFailStore:
        def close(self) -> None:
            events.append("close")
            raise SystemExit("store system exit")

    class BaseFailAgent:
        store = BaseFailStore()

        def init_session(self, label: str) -> None:
            events.append(f"init:{label}")

        def end_session(self) -> None:
            events.append("end")
            raise KeyboardInterrupt()
        def perceive(self, text: str) -> dict[str, str]:
            events.append("perceive")
            raise SystemExit("primary system exit")

    monkeypatch.setattr(commands_module, "_get_agent", lambda ctx: BaseFailAgent())
    result = CliRunner().invoke(cli, ["--offline", "quickstart"])
    assert isinstance(result.exception, SystemExit)
    assert "primary system exit" in str(result.exception)
    assert "store system exit" not in str(result.exception)
    assert "close" in events
    assert events.count("end") == 1


def test_offline_selection_does_not_replace_configured_production_model() -> None:
    config = MemoryAgentConfig.default()
    config.embedding.model_name = "my-production-model"
    config.embedding.provider = "sentence-transformers"

    runner = CliRunner()
    result = runner.invoke(cli, ["--offline", "--model", config.embedding.model_name, "stats"])

    assert result.exit_code == 0, result.output
    assert config.embedding.model_name == "my-production-model"
    assert config.embedding.provider == "sentence-transformers"
