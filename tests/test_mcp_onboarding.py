from __future__ import annotations

import json
import os
from pathlib import Path

from click.testing import CliRunner

from memory_agent.cli.commands import cli
from memory_agent.core.user_config import RetentionPolicy, load_user_config
from memory_agent.integrations.mcp_clients import ClientConfig, alfredo_server_spec, apply_proposal, discover_clients


def test_discovery_is_read_only_and_reports_known_clients(tmp_path: Path) -> None:
    config = tmp_path / "Claude" / "claude_desktop_config.json"
    config.parent.mkdir()
    original = '{"mcpServers":{"other":{"command":"other"}}}'
    config.write_text(original, encoding="utf-8")
    clients = discover_clients(platform="win32", appdata=tmp_path, home=tmp_path, cwd=tmp_path)
    assert config.read_text(encoding="utf-8") == original
    assert {client.name for client in clients} >= {"claude-desktop", "cursor", "vscode", "hermes"}


def test_proposal_preserves_unrelated_servers_and_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "claude.json"
    config.write_text('{"mcpServers":{"other":{"command":"other"}}}', encoding="utf-8")
    client = ClientConfig("claude-desktop", config, "json", "mcpServers", True, True, False)
    first = apply_proposal(client, alfredo_server_spec("python"), backup=True)
    second = apply_proposal(client, alfredo_server_spec("python"), backup=True)
    payload = json.loads(config.read_text(encoding="utf-8"))
    assert first.changed is True and first.backup_path is not None
    assert second.changed is False
    assert payload["mcpServers"]["other"] == {"command": "other"}
    assert payload["mcpServers"]["alfredo-memory"]["args"] == ["-m", "memory_agent", "mcp"]


def test_failed_replace_preserves_original(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "claude.json"
    original = b'{"mcpServers":{}}'
    config.write_bytes(original)
    monkeypatch.setattr(os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))
    client = ClientConfig("claude-desktop", config, "json", "mcpServers", True, True, False)
    change = apply_proposal(client, alfredo_server_spec("python"), backup=False)
    assert change.changed is False
    assert "replace failed" in (change.error or "")
    assert config.read_bytes() == original


def test_setup_persists_non_secret_preferences(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALFREDO_HOME", str(tmp_path))
    result = CliRunner().invoke(cli, ["setup"], input="n\ndays\n14\nteam-a\nuser-1\n")
    assert result.exit_code == 0, result.output
    config = load_user_config(tmp_path)
    assert config.onboarding_complete is True
    assert config.default_namespace == "team-a"
    assert config.default_user_id == "user-1"
    assert config.retention == RetentionPolicy(kind="days", days=14)
    assert "api_key" not in (tmp_path / "config.json").read_text(encoding="utf-8").lower()
