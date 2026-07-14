# Changelog

This file records the current project version and changes observable in this checkout. It does not claim that the package has already been published to PyPI.

## 0.2.0 (current project version)

The repository currently provides:

- the `alfredo-memory-agent` distribution metadata with the compatible `memory_agent` import namespace and `alfredo` CLI entry point;
- a deterministic offline quickstart and local SQLite memory lifecycle;
- namespace-aware retrieval, trust evidence, bounded context packets with selected and dropped IDs, reinforcement, supersession, decay, archive, and explicit forget operations;
- MCP stdio/HTTP integration through the optional `mcp` extra; the current default server embedding provider additionally requires the `semantic` extra (`[mcp,semantic]`).
- checked-in synthetic Alfredo Vault benchmark fixtures and offline comparison commands;
- public README, integration, architecture, security, contributing, roadmap, and community-policy documentation contracts.

The version is read from `pyproject.toml`; no separate unreleased or future release line is asserted here. Release dates, distribution availability, and compatibility guarantees beyond the documented APIs are not inferred from this checkout.
