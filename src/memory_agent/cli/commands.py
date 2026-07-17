"""Click-based CLI for MemoryAgent."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click

from memory_agent.core.config import MemoryAgentConfig
from memory_agent.core.embeddings import create_embedding_engine
from memory_agent.core.memory_store import MemoryStore
from memory_agent.integrations.markdown_export import export_markdown

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from memory_agent.agent.orchestrator import MemoryAgent


@click.group()
@click.option("--db", default=None, help="Path to SQLite database file")
@click.option(
    "--offline",
    is_flag=True,
    help="Use deterministic hashed-token embeddings without model downloads",
)
@click.option(
    "--model",
    default=None,
    help="Embedding model name (sentence-transformers)",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None, model: str | None, offline: bool) -> None:
    """Alfredo MemoryAgent — persistent memory for AI agents.

    An agent that accumulates experience autonomously, remembers
    user preferences, and retrieves critical memories within
    limited context windows.
    """
    ctx.ensure_object(dict)

    config = MemoryAgentConfig.default()
    if db:
        db_path = Path(db).resolve()
    else:
        db_path = Path(config.db_path).expanduser().resolve()
    if model:
        config.embedding.model_name = model
    if offline:
        config.embedding.provider = "deterministic"

    # Subcommands create MemoryAgent lazily. Benchmark commands can seed/evaluate
    # SQLite without loading an embedding model.
    ctx.obj["config"] = config
    ctx.obj["db_path"] = db_path
    ctx.obj["db_explicit"] = db is not None
    ctx.obj["agent"] = None


def _get_agent(ctx: click.Context) -> MemoryAgent:
    agent = ctx.obj.get("agent")
    if agent is None:
        from memory_agent.agent.orchestrator import MemoryAgent

        config = ctx.obj["config"]
        embedder = create_embedding_engine(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
            dimension=config.embedding.dimension,
            cache_size=config.embedding.cache_size,
        )
        agent = MemoryAgent(
            config=config,
            db_path=ctx.obj["db_path"],
            embedder=embedder,
        )
        ctx.obj["agent"] = agent
    return agent


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

@cli.command()
@click.pass_context
def quickstart(ctx: click.Context) -> None:
    """Run Alfredo MemoryAgent's offline SQLite cross-turn recall demo."""
    config = ctx.obj["config"]
    if config.embedding.provider != "deterministic":
        raise click.UsageError("quickstart requires the explicit --offline option")

    original_db_path = ctx.obj["db_path"]
    use_temporary_db = not ctx.obj.get("db_explicit", False)
    temporary = tempfile.TemporaryDirectory(prefix="memory-agent-quickstart-")
    agent = None
    session_active = False
    try:
        if use_temporary_db:
            ctx.obj["db_path"] = Path(temporary.name) / "memory.db"
        agent = _get_agent(ctx)
        agent.init_session(label="quickstart")
        session_active = True
        agent.perceive("I prefer Python for automation")
        agent.end_session()
        session_active = False

        agent.init_session(label="quickstart-recall")
        session_active = True
        result = agent.perceive("What programming language do I prefer?")
        recalled = result["recollection_text"]
        click.echo("Alfredo MemoryAgent — offline quickstart")
        click.echo("Persistent memory • timely forgetting • bounded, explainable recall")
        if not recalled:
            raise click.ClickException(
                "Quickstart could not recall the stored memory from SQLite"
            )
        click.echo("Remembered:")
        click.echo(recalled)
        click.echo(f"SQLite vault: {ctx.obj['db_path']}")
    finally:
        primary_active = sys.exc_info()[0] is not None
        cleanup_error: BaseException | None = None

        def run_cleanup(action) -> None:
            nonlocal cleanup_error
            try:
                action()
            except BaseException as exc:
                logger.debug("quickstart cleanup step failed", exc_info=True)
                if cleanup_error is None:
                    cleanup_error = exc

        if agent is not None and session_active:
            run_cleanup(agent.end_session)
        if agent is not None:
            run_cleanup(agent.store.close)
        run_cleanup(lambda: ctx.obj.__setitem__("db_path", original_db_path))
        run_cleanup(temporary.cleanup)
        if not primary_active and cleanup_error is not None:
            raise cleanup_error


@cli.command()
@click.option("--label", "-l", default="", help="Session label")
@click.pass_context
def chat(ctx: click.Context, label: str) -> None:
    """Start an interactive chat session."""
    agent = _get_agent(ctx)

    print("\n  MemoryAgent — Interactive Session")
    print("  Commands: /stats, /memories, /search <q>, /forget <id>, /help, /quit")
    print("=" * 50)

    agent.init_session(label=label or None)

    try:
        while True:
            try:
                user_input = input("\n  You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                _handle_command(agent, user_input)
                continue

            # Process through memory cycle
            result = agent.perceive(user_input)

            # Print recollections
            if result["recollection_text"]:
                print(f"\n  {'─' * 40}")
                print(result["recollection_text"])
                print(f"  {'─' * 40}")

            # Print new memories
            if result["new_memories"]:
                for mem in result["new_memories"]:
                    print(f"  [+] {mem.memory_type}: {mem.content[:60]}...")

            # Print stats line
            if result["archived"] > 0:
                print(f"  [archived: {result['archived']}]")

            # Generate a response
            response = _generate_response(agent, user_input, result)
            print(f"\n  Agent > {response}")

    finally:
        agent.end_session()
        print("\n  Session ended.")


def _handle_command(agent: MemoryAgent, cmd: str) -> None:
    """Handle a slash command."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/quit":
        raise KeyboardInterrupt()

    elif command == "/help":
        print("""
  Commands:
    /stats        — Show memory statistics
    /memories     — List all active memories
    /search <q>   — Semantic search memories
    /forget <id>  — Delete a memory by ID
    /help         — This help
    /quit         — Exit session
        """.strip())

    elif command == "/stats":
        stats = agent.get_stats()
        print(f"\n  Memory Statistics:")
        print(f"  {'─' * 40}")
        print(f"  Active memories:   {stats['total_active']}")
        print(f"  Archived:          {stats['archived']}")
        print(f"  Embeddings:        {stats['embedding_count']}")
        print(f"  Session turns:     {stats['session_turns']}")
        print(f"  Avg importance:    {stats['avg_importance']}")
        print(f"\n  By type:")
        for t, c in stats["type_distribution"].items():
            print(f"    {t}: {c}")
        print(f"\n  Decay lifespans:")
        for level, days in stats["decay_lifespans_days"].items():
            print(f"    {level}: ~{days} days")

    elif command == "/memories":
        memories = agent.list_memories()
        if not memories:
            print("  No memories.")
        namespace = getattr(agent, "namespace", None)
        print(f"  Namespace: {namespace}")
        print("  Lifecycle: active")
        print(f"  Selected IDs: {[m.id for m in memories[:20]]}")
        print("  Dropped IDs: []")
        for memory in memories[:20]:
            evidence = agent.explain_memory(memory)
            print(
                f"  Evidence #{memory.id}: "
                f"trust={evidence.get('trust', 'unknown')} "
                f"reason={evidence.get('reason', 'active memory')}"
            )
        print(f"\n  Active memories ({len(memories)}):")
        print(f"  {'─' * 40}")
        for m in memories[:20]:
            tags = f" [{', '.join(m.tags[:2])}]" if m.tags else ""
            print(f"  #{m.id}: [{m.memory_type}] {m.content[:70]}{tags}")
            print(f"       importance={m.importance:.1f} strength={m.strength:.2f} "
                  f"accesses={m.access_count}")

    elif command == "/search":
        if not arg:
            print("  Use: /search <query>")
            return
        payload = agent.search_memories(arg, top_k=5)
        results = payload.get("results", [])
        if not results:
            print("  No results.")
        print(f"\n  Search results for '{arg}':")
        print(f"  {'─' * 40}")
        for item in results:
            memory = item.get("memory", item)
            memory_type = memory.get("memory_type", memory.get("type", "memory"))
            print(f"  #{memory.get('id')} [{memory_type}] "
                  f"(score={float(item.get('score', 0.0)):.3f})")
            print(f"       {memory.get('content', '')[:80]}")
        lifecycle = payload.get("lifecycle", {})
        lifecycle_status = (
            lifecycle.get("status", "searched")
            if isinstance(lifecycle, dict)
            else lifecycle
        )
        print(f"  Namespace: {payload.get('namespace')}")
        print(f"  Lifecycle: {lifecycle_status}")
        print(f"  Selected IDs: {payload.get('selected_ids', [])}")
        print(f"  Dropped IDs: {payload.get('dropped_ids', [])}")
        for evidence in payload.get("evidence", []):
            print(
                f"  Evidence #{evidence.get('id')}: "
                f"trust={evidence.get('trust', 'unknown')} "
                f"reason={evidence.get('reason', '')}"
            )

    elif command == "/forget":
        if not arg or not arg.isdigit():
            print("  Use: /forget <memory_id>")
            return
        mid = int(arg)
        result = agent.forget_memory(mid)
        print(f"  Memory #{mid} {result.get('status', 'archived')}.")
        print(f"  Namespace: {result.get('namespace')}")
        print(f"  Lifecycle: {result.get('lifecycle', 'archived')}")
        print(f"  Selected IDs: {result.get('selected_ids', [])}")
        print(f"  Dropped IDs: {result.get('dropped_ids', [mid])}")
        print(f"  Trust: {result.get('trust', 'unknown')}")
        print(f"  Reason: {result.get('reason', '')}")

    else:
        print(f"  Unknown command: {command}. Type /help")


def _generate_response(
    agent: MemoryAgent, user_input: str, result: dict
) -> str:
    """Generate a response based on the user input and recollections.

    In a full deployment, this would call an LLM. For this demo,
    we use a template-based response.
    """
    recollections = result["recollection_text"]
    new_count = len(result["new_memories"])
    total = result["total_memories"]

    # Simple response template
    response_parts = []

    # Acknowledge new memories
    if new_count > 0:
        response_parts.append(
            f"Understood. I stored {new_count} "
            f"{'new memory' if new_count == 1 else 'new memories'}."
        )

    # Reference recollections
    if recollections:
        top = result["recollections"][0]
        response_parts.append(
            f"I remembered that {top.memory.content[:50].lower()}..."
        )

    # Final note
    response_parts.append(
        f"I now have {total} {'memory' if total == 1 else 'memories'} in persistent storage."
    )

    return " ".join(response_parts)


# ------------------------------------------------------------------
# Additional CLI commands
# ------------------------------------------------------------------


@cli.command()
@click.option("--namespace", default=None, help="Memory namespace")
@click.pass_context
def stats(ctx: click.Context, namespace: str | None) -> None:
    """Show namespace-scoped memory statistics without starting a session."""
    agent = _get_agent(ctx)
    stats_data = agent.get_stats(namespace=namespace)

    click.echo("\n📊 MemoryAgent Statistics")
    click.echo("━" * 40)
    click.echo(f"  Namespace:          {stats_data.get('namespace', namespace)}")
    click.echo(f"  Lifecycle:          {stats_data.get('lifecycle', 'active')}")
    if "trust" in stats_data:
        click.echo(f"  Trust:              {stats_data['trust']}")
    if "reason" in stats_data:
        click.echo(f"  Reason:             {stats_data['reason']}")
    click.echo(f"  Active memories:   {stats_data['total_active']}")
    click.echo(f"  Archived:          {stats_data['archived']}")
    click.echo(f"  Embeddings:        {stats_data['embedding_count']}")
    click.echo(f"  Avg importance:    {stats_data['avg_importance']}")
    click.echo()
    click.echo("  By type:")
    for t, c in stats_data["type_distribution"].items():
        click.echo(f"    {t}: {c}")
    click.echo()
    click.echo("  Decay lifespans:")
    for level, days in stats_data["decay_lifespans_days"].items():
        click.echo(f"    {level}: ~{days} days")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results")
@click.option("--namespace", default=None, help="Memory namespace")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, namespace: str | None) -> None:
    """Semantic memory search through the MemoryAgent facade."""
    agent = _get_agent(ctx)
    payload = agent.search_memories(query, top_k=top_k, namespace=namespace)
    results = payload.get("results", [])

    click.echo(f"\n  Search results for: '{query}'")
    click.echo(f"  Namespace: {payload.get('namespace', namespace)}")
    click.echo(f"  Lifecycle: {payload.get('lifecycle', {}).get('status', 'searched') if isinstance(payload.get('lifecycle'), dict) else payload.get('lifecycle', 'searched')}")
    click.echo("━" * 50)
    if not results:
        click.echo("  No matching memories found.")
    for i, item in enumerate(results, 1):
        memory = item.get("memory", item)
        memory_type = memory.get("memory_type", memory.get("type", "memory"))
        score = item.get("score", 0.0)
        trust = item.get("trust") or item.get("evidence", {}).get("trust", "unknown")
        reason = item.get("reason") or item.get("evidence", {}).get("reason", "")
        click.echo(
            f"  {i}. #{memory.get('id')} [{memory_type}] "
            f"(score={float(score):.3f}, imp={float(memory.get('importance', 0.0)):.1f}, trust={trust})"
        )
        click.echo(f"     {memory.get('content', '')[:80]}")
        if reason:
            click.echo(f"     Reason: {reason}")
        click.echo()
    click.echo(f"  Selected IDs: {payload.get('selected_ids', [])}")
    click.echo(f"  Dropped IDs: {payload.get('dropped_ids', [])}")
    for evidence in payload.get("evidence", []):
        click.echo(
            f"  Evidence #{evidence.get('id')}: trust={evidence.get('trust', 'unknown')} "
            f"reason={evidence.get('reason', '')}"
        )




@cli.command()
@click.argument("memory_id", type=int)
@click.option("--namespace", default=None, help="Memory namespace")
@click.pass_context
def forget(ctx: click.Context, memory_id: int, namespace: str | None) -> None:
    """Archive a memory through the MemoryAgent facade."""
    agent = _get_agent(ctx)
    payload = agent.forget_memory(memory_id, namespace=namespace)
    click.echo(f"  Memory #{memory_id}: {payload.get('status', 'archived')}")
    click.echo(f"  Namespace: {payload.get('namespace', namespace)}")
    click.echo(f"  Lifecycle: {payload.get('lifecycle', 'archived')}")
    click.echo(f"  Trust: {payload.get('trust', 'unknown')}")
    click.echo(f"  Reason: {payload.get('reason', '')}")

@cli.command()
@click.option("--namespace", default=None, help="Memory namespace")
@click.pass_context
def memories(ctx: click.Context, namespace: str | None) -> None:
    """List all active memories through the MemoryAgent facade."""
    agent = _get_agent(ctx)
    all_memories = agent.list_memories(namespace=namespace)

    if not all_memories:
        click.echo("  No memories stored.")
        return

    click.echo(f"\n  Active Memories ({len(all_memories)})")
    click.echo("━" * 50)
    for m in all_memories:
        click.echo(
            f"  #{m.id:<4} [{m.memory_type:<10}] "
            f"imp={m.importance:.2f} str={m.strength:.2f} "
            f"acc={m.access_count} namespace={m.namespace}"
        )
        click.echo(f"      {m.content[:80]}")
        click.echo()


@cli.command("export-markdown")
@click.option("--namespace", required=True, help="Exact memory namespace to export")
@click.option(
    "--output",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for the Markdown projection",
)
@click.pass_context
def export_markdown_command(ctx: click.Context, namespace: str, output: Path) -> None:
    """Export active memories from SQLite as deterministic Markdown files."""
    store = MemoryStore(ctx.obj["db_path"])
    try:
        store.initialize()
        export_markdown(store, output, namespace)
    finally:
        store.close()
    click.echo(f"Exported Markdown memories to {output}")



@cli.group()
def benchmark() -> None:
    """Run Alfredo's Vault synthetic memory benchmark."""


@benchmark.command("seed")
@click.option(
    "--users",
    "users_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to USERS_JSON.",
)
@click.option(
    "--memories",
    "memories_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to MEMORIES_JSONL.",
)
@click.option("--expected-users", type=int, default=None, help="Exact expected user count.")
@click.option("--expected-memories", type=int, default=None, help="Exact expected memory count.")
@click.pass_context
def benchmark_seed(
    ctx: click.Context,
    users_path: Path,
    memories_path: Path,
    expected_users: int | None,
    expected_memories: int | None,
) -> None:
    """Load benchmark users and memories into the configured SQLite vault."""
    from memory_agent.benchmark import (
        load_memories_jsonl,
        load_users,
        seed_memory_store,
    )

    db_path = ctx.obj["db_path"]
    store = MemoryStore(db_path)
    store.initialize()
    try:
        users = load_users(users_path, expected_count=expected_users)
        memories = load_memories_jsonl(memories_path, expected_count=expected_memories)
        stats = seed_memory_store(store, users, memories)
    finally:
        store.close()
    click.echo("ALFREDO VAULT BENCHMARK SEEDED")
    click.echo(f"DB: {db_path}")
    click.echo(f"Users: {len(users)}")
    click.echo(f"Memories inserted: {stats['inserted']}")
    click.echo(f"Active memories: {stats['active']}")
    click.echo(f"Inactive memories: {stats['inactive']}")


@benchmark.command("run")
@click.option(
    "--users",
    "users_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to USERS_JSON.",
)
@click.option(
    "--questions",
    "questions_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to EVALUATION_QUESTIONS_JSONL.",
)
@click.option(
    "--report",
    "report_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write benchmark_report.json.",
)
@click.option("--expected-users", type=int, default=None, help="Exact expected user count.")
@click.option("--expected-questions", type=int, default=None, help="Exact expected question count.")
@click.pass_context
def benchmark_run(
    ctx: click.Context,
    users_path: Path,
    questions_path: Path,
    report_path: Path,
    expected_users: int | None,
    expected_questions: int | None,
) -> None:
    """Evaluate benchmark questions against the configured SQLite vault."""
    from memory_agent.benchmark import (
        evaluate_questions,
        load_questions_jsonl,
        load_users,
        write_report,
    )

    db_path = ctx.obj["db_path"]
    store = MemoryStore(db_path)
    store.initialize()
    try:
        users = load_users(users_path, expected_count=expected_users)
        questions = load_questions_jsonl(questions_path, expected_count=expected_questions)
        results = evaluate_questions(store, questions, users=users)
    finally:
        store.close()
    report = write_report(results, report_path)
    metrics = report["metrics"]
    click.echo("ALFREDO VAULT BENCHMARK REPORT")
    click.echo(f"DB: {db_path}")
    click.echo(f"Questions: {metrics['total_questions']}")
    click.echo(f"Passed: {metrics['passed']}")
    click.echo(f"Failed: {metrics['failed']}")
    click.echo(f"Accuracy: {metrics['accuracy_percentage']:.2f}%")
    click.echo(f"Security events: {metrics['security_events']}")
    click.echo(f"Report: {report_path}")


@benchmark.command("compare")
@click.option("--offline", "offline_requested", is_flag=True, help="Use deterministic offline baselines without external APIs.")
@click.option("--users", "users_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to USERS_JSON.")
@click.option("--memories", "memories_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to MEMORIES_JSONL.")
@click.option("--questions", "questions_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to EVALUATION_QUESTIONS_JSONL.")
@click.option("--report", "report_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Where to write comparison report JSON.")
@click.option("--seed", type=int, default=0, show_default=True, help="Reproducibility seed.")
@click.option("--run", "run_id", default="local", show_default=True, help="Stable run identifier.")
@click.option("--expected-users", type=int, default=None, help="Exact expected user count.")
@click.option("--expected-memories", type=int, default=None, help="Exact expected memory count.")
@click.option("--expected-questions", type=int, default=None, help="Exact expected question count.")
@click.pass_context
def benchmark_compare(
    ctx: click.Context,
    users_path: Path,
    memories_path: Path,
    questions_path: Path,
    report_path: Path,
    seed: int,
    run_id: str,
    offline_requested: bool,
    expected_users: int | None,
    expected_memories: int | None,
    expected_questions: int | None,
) -> None:
    """Compare all three offline memory retrieval strategies."""
    configured_offline = ctx.obj.get("config").embedding.provider == "deterministic"
    if not offline_requested and not configured_offline:
        raise click.UsageError("benchmark compare requires explicit --offline")
    from memory_agent.benchmark import compare_benchmarks, load_memories_jsonl, load_questions_jsonl, load_users

    users = load_users(users_path, expected_count=expected_users)
    memories = load_memories_jsonl(memories_path, expected_count=expected_memories)
    questions = load_questions_jsonl(questions_path, expected_count=expected_questions)
    report = compare_benchmarks(
        users,
        memories,
        questions,
        seed=seed,
        run_id=run_id,
        offline=bool(offline_requested or ctx.obj.get("config").embedding.provider == "deterministic"),
        report_path=report_path,
    )
    click.echo("ALFREDO VAULT BENCHMARK COMPARISON")
    click.echo(f"Strategies: {', '.join(report['strategies'])}")
    click.echo(f"Seed: {report['seed']}")
    click.echo(f"Run: {report['run_id']}")
    click.echo(f"Report: {report_path}")


@cli.command()
@click.argument("action", required=False, default="serve", type=click.Choice(["serve", "setup"], case_sensitive=False))
@click.option("--http", is_flag=True, help="Run in HTTP mode instead of stdio")
@click.option("--host", default="localhost", help="HTTP host (default: localhost)")
@click.option("--port", default=8090, type=int, help="HTTP port (default: 8090)")
@click.pass_context
def mcp(ctx: click.Context, action: str, http: bool, host: str, port: int) -> None:
    """Run Alfredo as an MCP server or configure MCP clients."""
    if action.lower() == "setup":
        from memory_agent.cli.onboarding import run_mcp_setup

        run_mcp_setup()
        return
    from memory_agent.integrations.mcp_server import run_mcp_server

    run_mcp_server(host=host if http else None, port=port if http else None)


@cli.command()
@click.option("--provider", "-p", default="qwencloud",
              help="LLM provider: qwencloud, deepseek, openrouter, openai, anthropic")
@click.option("--model", "-m", default=None, help="Model name override")
@click.option("--query", "-q", default=None, help="Single query (non-interactive)")
@click.pass_context
def llm(ctx: click.Context, provider: str, model: str | None, query: str | None) -> None:
    """Chat with Qwen Cloud or another LLM using MemoryAgent memory.

    Requires the corresponding API key env var to be set.
    """
    from memory_agent.integrations.llm_connector import run_interactive, LLMConnector

    if query:
        connector = LLMConnector(
            provider=provider, model=model,
            db_path=ctx.obj.get("db_path"),
            system_prompt="You are a helpful assistant with persistent memory.",
        )
        connector.agent.init_session("llm-single")
        response = connector.turn(query)
        print(response)
        connector.close()
    else:
        run_interactive(provider=provider, model=model)


@cli.command("setup")
def setup_command() -> None:
    """Run first-run setup and configure detected MCP clients."""
    from memory_agent.cli.onboarding import run_setup

    run_setup()


@cli.command("doctor")
@click.option("--mcp", "mcp_only", is_flag=True, help="Only inspect MCP clients.")
def doctor_command(mcp_only: bool) -> None:
    """Diagnose paths, dependencies, and MCP integrations."""
    from memory_agent.cli.onboarding import run_doctor

    run_doctor(mcp_only=mcp_only)


@cli.command("version")
def version_command() -> None:
    """Print the installed Alfredo version."""
    from importlib.metadata import version

    click.echo(version("alfredo-memory-agent"))
