"""Tests for Alfredo native memory paths."""

from __future__ import annotations

from pathlib import Path

from memory_agent.core.paths import default_memory_db_path, resolve_memory_home
from click.testing import CliRunner

from memory_agent.cli.commands import cli
from memory_agent.core.config import MemoryAgentConfig


def test_resolve_memory_home_uses_alfredo_home_env(monkeypatch, tmp_path):
    home = tmp_path / "custom-vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    resolved = resolve_memory_home(project_root=tmp_path / "repo")

    assert resolved == home
    assert resolved.exists()


def test_resolve_memory_home_uses_repo_local_alfredo_for_dev_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("ALFREDO_HOME", raising=False)
    repo = tmp_path / "Alfredo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "alfredo-memory-agent"\n', encoding="utf-8")

    resolved = resolve_memory_home(project_root=repo)

    assert resolved == repo / ".alfredo"
    assert resolved.exists()


def test_default_memory_db_path_uses_memory_agent_db_filename(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    db_path = default_memory_db_path()

    assert db_path.name == "memory_agent.db"
    assert db_path.parent == home



def test_memory_agent_config_default_uses_native_db(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    config = MemoryAgentConfig.default()

    assert Path(config.db_path) == home / "memory_agent.db"


def test_cli_stats_uses_native_default_db(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))
    runner = CliRunner()

    result = runner.invoke(cli, ["stats"])

    assert result.exit_code == 0
    assert (home / "memory_agent.db").exists()