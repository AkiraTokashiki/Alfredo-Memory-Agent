"""Interactive setup for Alfredo's local CLI and MCP integrations."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import click

from memory_agent.core.capabilities import detect_capabilities, missing_capability_text
from memory_agent.core.paths import resolve_memory_home
from memory_agent.core.user_config import AlfredoUserConfig, RetentionPolicy, load_user_config, save_user_config
from memory_agent.integrations.mcp_clients import apply_proposal, alfredo_server_spec, discover_clients, manual_snippet


def _mcp_clients() -> tuple:
    appdata = Path(os.environ["APPDATA"]) if os.environ.get("APPDATA") else None
    return discover_clients(appdata=appdata, home=Path.home(), cwd=Path.cwd())


def print_capabilities() -> None:
    for status in detect_capabilities():
        click.echo(f"  {status.name}: {'available' if status.available else 'missing'} — {status.detail}")
        if not status.available:
            click.echo(f"    {missing_capability_text(status)}")


def run_mcp_setup() -> None:
    """Inspect clients and apply only explicitly confirmed JSON changes."""
    click.echo("Alfredo MCP client setup")
    print_capabilities()
    spec = alfredo_server_spec()
    for client in _mcp_clients():
        if not client.exists:
            click.echo(f"  {client.name}: not detected ({client.path})")
            continue
        if client.format == "manual":
            click.echo(f"  {client.name}: manual configuration")
            click.echo(manual_snippet(client, spec))
            continue
        if not click.confirm(f"  Configure {client.name} at {client.path}?", default=False):
            click.echo("    Skipped.")
            continue
        change = apply_proposal(client, spec, backup=True)
        if change.error:
            click.echo(f"    Failed: {change.error}")
        elif change.changed:
            suffix = f" Backup: {change.backup_path}" if change.backup_path else ""
            click.echo(f"    Configured.{suffix}")
        else:
            click.echo("    Already configured.")


def run_setup() -> AlfredoUserConfig:
    """Run setup and persist only validated non-secret preferences."""
    home = resolve_memory_home()
    current = load_user_config(home)
    click.echo("Alfredo first-run setup")
    print_capabilities()
    if click.confirm("Configure detected MCP clients now?", default=True):
        run_mcp_setup()
    kind = click.prompt("Session history retention", default=current.retention.kind, type=click.Choice(["forever", "days", "none"]))
    days = None
    if kind == "days":
        while days is None:
            raw = click.prompt("Delete ended sessions after how many days", default=current.retention.days or 30)
            try:
                candidate = int(raw)
                days = candidate if candidate > 0 else None
            except ValueError:
                days = None
            if days is None:
                click.echo("Retention days must be a positive integer.")
    namespace = click.prompt("Default namespace (blank for default)", default=current.default_namespace or "", show_default=False).strip() or None
    user_id = click.prompt("Default user ID (blank for none)", default=current.default_user_id or "", show_default=False).strip() or None
    updated = AlfredoUserConfig(
        onboarding_complete=True,
        default_namespace=namespace,
        default_user_id=user_id,
        retention=RetentionPolicy(kind=kind, days=days),
        mcp_clients=current.mcp_clients,
    )
    path = save_user_config(updated, home)
    click.echo(f"Configuration saved to {path}")
    return updated


def run_doctor(mcp_only: bool = False) -> None:
    click.echo(f"Alfredo home: {resolve_memory_home()}")
    if not mcp_only:
        print_capabilities()
    for client in _mcp_clients():
        state = "present" if client.exists else "not detected"
        writable = "writable" if client.writable else "not writable"
        click.echo(f"{client.name}: {state}, {writable}, path={client.path}")


def enforce_retention(agent, policy: RetentionPolicy) -> int:
    if policy.kind == "forever":
        return 0
    cutoff = datetime.now().isoformat()
    if policy.kind == "days":
        cutoff = (datetime.now() - timedelta(days=policy.days or 0)).isoformat()
    return agent.store.purge_expired_sessions(cutoff=cutoff, namespace=getattr(agent, "namespace", None))
