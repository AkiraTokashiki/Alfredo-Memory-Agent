"""Tests for Alfredo native memory paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_agent.core.paths import _is_dev_repo
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



@pytest.mark.parametrize(
    "document",
    [
        '[project]\nname = "alfredo-memory-agent"\n',
        '# comment\n[project]\nversion = "0.2.0"\nname = "alfredo-memory-agent"\n',
    ],
)
def test_is_dev_repo_parses_project_name_from_toml(document, tmp_path):
    (tmp_path / "pyproject.toml").write_text(document, encoding="utf-8")

    assert _is_dev_repo(tmp_path) is True


@pytest.mark.parametrize(
    "document",
    [
        "[project]\nname = \"alfredo-memory-agent\"\ninvalid",
        '[tool]\nname = "alfredo-memory-agent"\n',
        '[project]\nname = "another-project"\n',
    ],
)
def test_is_dev_repo_rejects_invalid_or_non_project_metadata(document, tmp_path):
    (tmp_path / "pyproject.toml").write_text(document, encoding="utf-8")

    assert _is_dev_repo(tmp_path) is False



def test_is_dev_repo_handles_unreadable_pyproject(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "alfredo-memory-agent"\n', encoding="utf-8"
    )

    def raise_os_error(*args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", raise_os_error)

    assert _is_dev_repo(tmp_path) is False


def test_is_dev_repo_handles_invalid_utf8_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname = \xff\n")

    assert _is_dev_repo(tmp_path) is False


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