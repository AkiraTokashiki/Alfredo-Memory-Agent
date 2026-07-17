"""Safe discovery and configuration helpers for local MCP clients."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class McpConfigError(ValueError):
    """Raised when an MCP client configuration cannot be safely processed."""


@dataclass(frozen=True)
class ClientConfig:
    name: str
    path: Path
    format: Literal["json", "manual"]
    servers_key: str
    exists: bool = False
    writable: bool = False
    alfredo_present: bool = False


@dataclass(frozen=True)
class McpServerSpec:
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class McpChange:
    client: ClientConfig
    changed: bool
    backup_path: Path | None = None
    manual_snippet: str | None = None
    error: str | None = None


def _candidate_paths(*, platform: str, appdata: Path | None, home: Path, cwd: Path):
    if platform == "win32":
        claude_root = appdata or home / "AppData" / "Roaming"
    elif platform == "darwin":
        claude_root = home / "Library" / "Application Support"
    else:
        claude_root = home / ".config"
    return (
        ("claude-desktop", claude_root / "Claude" / "claude_desktop_config.json", "json", "mcpServers"),
        ("cursor", cwd / ".cursor" / "mcp.json", "json", "mcpServers"),
        ("cursor-user", home / ".cursor" / "mcp.json", "json", "mcpServers"),
        ("vscode", cwd / ".vscode" / "mcp.json", "json", "servers"),
        ("hermes", home / ".hermes" / "config.yaml", "manual", "mcp_servers"),
    )


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise McpConfigError(f"Could not read MCP configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise McpConfigError(f"MCP configuration {path} must contain a JSON object")
    return value


def discover_clients(*, platform: str | None = None, appdata: Path | None = None, home: Path | None = None, cwd: Path | None = None) -> tuple[ClientConfig, ...]:
    platform_name = platform or sys.platform
    home_path = (home or Path.home()).expanduser().resolve()
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    clients: list[ClientConfig] = []
    for name, path, file_format, servers_key in _candidate_paths(platform=platform_name, appdata=appdata, home=home_path, cwd=cwd_path):
        exists = path.is_file()
        alfredo_present = False
        if exists and file_format == "json":
            document = _read_json(path)
            servers = document.get(servers_key) if document else None
            alfredo_present = isinstance(servers, dict) and "alfredo-memory" in servers
        clients.append(ClientConfig(name, path, file_format, servers_key, exists, path.parent.exists() and os.access(path.parent, os.W_OK), alfredo_present))
    return tuple(clients)


def alfredo_server_spec(python_executable: str | None = None) -> McpServerSpec:
    return McpServerSpec(python_executable or sys.executable, ("-m", "memory_agent", "mcp"), {})


def _payload(spec: McpServerSpec) -> dict[str, object]:
    return {"command": spec.command, "args": list(spec.args), "env": dict(spec.env)}


def manual_snippet(client: ClientConfig, spec: McpServerSpec) -> str:
    if client.name == "hermes":
        return "mcp_servers:\n  alfredo-memory:\n    command: " + spec.command + "\n    args: [\"-m\", \"memory_agent\", \"mcp\"]"
    return json.dumps({"alfredo-memory": _payload(spec)}, indent=2)


def _backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{index}.bak")
        index += 1
    return candidate


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def apply_proposal(client: ClientConfig, spec: McpServerSpec, *, backup: bool = True) -> McpChange:
    if client.format != "json":
        return McpChange(client, False, manual_snippet=manual_snippet(client, spec))
    try:
        document = _read_json(client.path) or {}
        servers = document.get(client.servers_key) or {}
        if not isinstance(servers, dict):
            raise McpConfigError(f"MCP configuration key {client.servers_key!r} must be an object")
        desired = _payload(spec)
        if servers.get("alfredo-memory") == desired:
            return McpChange(client, False)
        document[client.servers_key] = {**servers, "alfredo-memory": desired}
        backup_path = None
        if client.path.exists() and backup:
            backup_path = _backup_path(client.path)
            backup_path.write_bytes(client.path.read_bytes())
        _write_atomic(client.path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        return McpChange(client, True, backup_path=backup_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, McpConfigError) as exc:
        return McpChange(client, False, error=str(exc))
