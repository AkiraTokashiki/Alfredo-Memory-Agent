"""Contract tests for MCP server transport startup."""

from __future__ import annotations

from memory_agent.integrations import mcp_server


def test_http_startup_configures_settings_and_uses_streamable_http_without_network(
    monkeypatch,
) -> None:
    """HTTP startup prewarms the agent and delegates only the transport to FastMCP."""
    warmed_agents: list[object] = []
    run_transports: list[str] = []
    run_settings: list[tuple[str, int]] = []
    fake_agent = object()

    def fake_get_agent() -> object:
        warmed_agents.append(fake_agent)
        return fake_agent

    def fake_run(*, transport: str) -> None:
        # A transport-only signature makes passing unsupported host/port kwargs fail.
        run_settings.append((mcp_server.mcp.settings.host, mcp_server.mcp.settings.port))
        run_transports.append(transport)

    monkeypatch.setattr(mcp_server, "_get_agent", fake_get_agent)
    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    settings = mcp_server.mcp.settings
    original_host = settings.host
    original_port = settings.port

    with monkeypatch.context() as settings_patch:
        settings_patch.setattr(settings, "host", "before.example")
        settings_patch.setattr(settings, "port", 8123)

        mcp_server.run_mcp_server("127.0.0.1", 9876)

        assert run_settings == [("127.0.0.1", 9876)]
        assert warmed_agents == [fake_agent]
        assert run_transports == ["streamable-http"]

    assert settings.host == original_host
    assert settings.port == original_port


def test_stdio_startup_prewarm_agent_and_uses_stdio_transport(monkeypatch) -> None:
    """Stdio startup prewarms the agent and selects the stdio transport."""
    warmed_agents: list[object] = []
    run_transports: list[str] = []
    fake_agent = object()

    def fake_get_agent() -> object:
        warmed_agents.append(fake_agent)
        return fake_agent

    def fake_run(*, transport: str) -> None:
        run_transports.append(transport)

    monkeypatch.setattr(mcp_server, "_get_agent", fake_get_agent)
    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    mcp_server.run_mcp_server()

    assert warmed_agents == [fake_agent]
    assert run_transports == ["stdio"]
