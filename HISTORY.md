# Project History — MemoryAgent

## 2026-07-03

### Completed features

- Previous foundation work.
- **MCP Server**: FastMCP server exposing six tools (`memory__perceive`, `memory__search`, `memory__store`, `memory__stats`, `memory__forget`, `memory__reinforce`), two resources, and one prompt. Supports stdio and Streamable HTTP at `/mcp`. Integrates with Hermes through `hermes mcp add`, and can also be used by Claude Desktop, Cursor, and other MCP clients.
- **LLM Connector**: `python -m memory_agent llm` provides an interactive session with Qwen Cloud, DeepSeek, OpenRouter, OpenAI, or Anthropic while using MemoryAgent as persistent memory. Qwen Cloud, DeepSeek, OpenRouter, and OpenAI use OpenAI-compatible chat completions; Anthropic uses its native Messages API. Five providers are supported.
- **Integration guide**: `INTEGRATION.md` documents Hermes configuration, MCP setup, terminal commands, and programmatic usage.
- **Memory Store**: SQLite with WAL mode, five tables (`memories`, `embeddings`, `memory_tags`, `sessions`, `session_memories`), full CRUD, soft/hard delete, batch operations, and keyword-search fallback.
- **Embedding Engine**: sentence-transformers (`all-MiniLM-L6-v2`, 384 dimensions), LRU cache, `encode_multiple` batching, and cosine similarity.
- **Forgetting Curve**: Ebbinghaus-style exponential decay modulated by importance. High importance lasts about 90 days, medium about 21 days, and low about 3 days. Each retrieval reinforces memory strength by `+0.15`. Low-strength memories are archived automatically.
- **Retrieval Engine**: Combined scoring: 40% semantic similarity, 20% recency, 20% importance, and 20% strength. MMR diversity penalty reduces semantic duplicates.
- **Decision Engine**: Natural-language extraction for preferences and facts. It accepts English and Spanish input, while public memories and responses are emitted in English.
- **Agent Orchestrator**: Full perceive → extract → retrieve → decay cycle. Supports multi-turn sessions, periodic consolidation, and session-memory links.
- **CLI**: Click-based interactive chat with `/stats`, `/memories`, `/search`, and `/forget` commands.
- **Tests**: Initial suite passed after compatibility fixes.
- **Demos**: `demo_basic.py` shows a single-session interaction. `demo_multi_session.py` shows persistence across three sessions and forgetting behavior.

### Fixes applied

- Replaced `sqlite3.Row.get()` with bracket access (`row["column"]`) for Python 3.14 compatibility.
- Updated sentence-transformers dimension lookup from `get_sentence_embedding_dimension()` to `get_embedding_dimension()`.
- Simplified `test_decay_is_exponential` to check the exponential relationship directly.
- Optimized retrieval tests that repeatedly created `EmbeddingEngine` instances by using batched encoding.
- Broadened preference extraction so simple preference statements match reliably.

### Known issues

- Preference extraction can overlap on complex phrasing. Retrieval scoring and consolidation reduce the practical impact.
- Template responses are used when no LLM connector is active. Production usage should connect a model.
- The first semantic retrieval may load the sentence-transformers model lazily.

### Technology stack

- Python 3.14.3
- SQLite from the standard library, using WAL mode
- sentence-transformers 5.6.0 (`all-MiniLM-L6-v2`)
- numpy, click, pytest
- Editable install through setuptools

## 2026-07-13

### SDK and benchmark release candidate

- Added public storage, embedding, retrieval and trust protocols with explainable evidence.
- Added idempotent SQLite migration, namespace isolation, provider/dimension checks and explicit transaction boundaries.
- Added deterministic offline embeddings and a five-minute local quickstart without API keys or model downloads.
- Hardened CLI/MCP adapters so namespace, lifecycle, selected/dropped IDs and trust reasons remain visible through the public facade.
- Added reproducible raw-history, semantic-RAG and Alfredo benchmark baselines with synthetic-only validation, supersession-aware scoring, security metrics, dataset/config hashes and latency/context aggregates.
- Verified targeted release checks: 166 tests passed; offline quickstart completed; offline benchmark comparison produced all three strategies.

### Scope notes

- The managed multi-tenant platform, dashboard and billing remain out of scope for this release candidate.
- Stars and adoption are not guaranteed by the implementation; distribution quality depends on documentation, integrations and community use.
