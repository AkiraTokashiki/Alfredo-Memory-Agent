# Alfredo — MemoryAgent

![Alfredo MemoryAgent lifecycle: learn, retrieve, trust, pack, reinforce, supersede or forget](docs/assets/alfredo-memory-lifecycle.svg)

> **A local MemoryAgent that learns, remembers, forgets, and explains what it recalls.**
> Alfredo keeps a durable, selective memory for an agent without turning your data into a hosted SaaS.

**Start here:** [offline quickstart](#quickstart) · [lifecycle demo](examples/demo_lifecycle.py) · [MCP and Python integration](INTEGRATION.md) · [synthetic benchmark](#benchmark-evidence)

---

## What Alfredo is

Alfredo is a Python SDK and CLI for a local memory layer. Its native store is SQLite, and the intended distribution name is `alfredo-memory-agent` while the import namespace remains `memory_agent`. You run it beside your agent and choose where its vault lives; there is no Alfredo-hosted dashboard, tenant service, billing account, or required remote memory API.

A conversation transcript is not a memory policy. Raw history grows without a bounded selection step, while a simple RAG index can return semantically similar text without deciding whether it is trusted, stale, superseded, or safe to pack into a prompt. Alfredo makes those decisions explicit and inspectable:

- **Namespaces** keep records and sessions scoped to an agent or tenant boundary.
- **Evidence** records score, matching signals, trust, and a reason for each retrieval decision.
- **Selected and dropped IDs** show what entered the bounded recall packet and what did not.
- **Lifecycle state** makes reinforcement, supersession, archival, and explicit forgetting visible.

These are SDK behaviors, not a promise of automatic compliance or production security. Configure storage, access control, retention, and deployment isolation for your environment.

## Quickstart

The canonical install contract uses the `alfredo-memory-agent` distribution name and the `memory_agent` module:

```bash
pip install alfredo-memory-agent
python -m memory_agent --offline quickstart
```

The offline command uses deterministic local embeddings and a temporary SQLite vault, so this path does not require an API key or a model download. For a persistent vault, pass an explicit `--db` path to the CLI commands or configure the SDK store in your application.

For semantic embeddings from a checkout, install the optional provider with double quotes (the form works in Windows shells as well as POSIX shells):

```powershell
pip install -e ".[semantic]"
python -m memory_agent chat
```

The repository's `requirements.txt` contains the equivalent editable requirement `-e .[semantic]`. Keep semantic and offline vaults separate when their embedding provider or vector dimension differs.

## Lifecycle: from learning to an explainable packet

Every turn can follow this bounded path:

1. **Learn** — extract a candidate preference, fact, or interaction from input and attach its namespace and provenance fields.
2. **Retrieve** — search active records and rank candidates using the configured retrieval signals.
3. **Trust** — evaluate confidence and attach evidence; untrusted candidates are not allowed to consume context budget.
4. **Pack** — build a bounded recall packet, exposing `selected_ids` and `dropped_ids` instead of copying the whole vault into a prompt.
5. **Reinforce** — useful recalled records can be reinforced, while forgetting decay reduces stale strength over time.
6. **Supersede / forget** — a changed preference can supersede an older record, and an explicit forget request archives matching records in the active namespace.

Run the deterministic walkthrough with:

```bash
python examples/demo_lifecycle.py
```

The demo covers cross-session recall, preference supersession, and bounded trusted context without network access, model downloads, API keys, or wall-clock output. See [the full integration guide](INTEGRATION.md) for MCP stdio/HTTP and programmatic usage.

## Why it is different

| Approach | What it keeps | What it does not decide | What Alfredo adds |
| --- | --- | --- | --- |
| Raw conversation history | The transcript | Which facts are current, trusted, or within budget | Candidate extraction, lifecycle state, bounded packets, and explicit forget/supersede operations |
| Simple semantic RAG | Indexed chunks ranked by similarity | Whether a result is low-confidence, stale, superseded, or safe for context | Trust evidence, namespace-aware retrieval, selected/dropped IDs, and reinforcement/decay |
| Alfredo MemoryAgent | Structured local memories in a SQLite vault | Your deployment's access-control and privacy obligations | A local, inspectable learn → retrieve → trust → pack → reinforce → supersede/forget lifecycle |

The comparison is about responsibilities, not a claim that one strategy wins every workload. Alfredo can be embedded through the protocols in the SDK and can also expose an MCP server; it is not a hosted memory SaaS.

## Architecture in brief

```text
agent input
    │
    ▼
MemoryAgent orchestrator ──► extraction/consolidation ──► namespace-scoped SQLite vault
    │                                  │                           │
    │                                  └─ reinforce/supersede/forget┘
    ▼
retrieval ──► trust policy + evidence ──► bounded recall packet
    │                                        (selected_ids, dropped_ids)
    ▼
agent context / response
```

The offline embedding engine is deterministic. The optional semantic provider is selected explicitly. Storage, embedding, retrieval, and trust boundaries are injectable through the SDK ports, while `INTEGRATION.md` documents the CLI, MCP, and Python entry points.

## Benchmark evidence

The [Alfredo's Vault fixtures](benchmarks/alfredos_vault/) are a **synthetic** comparison dataset for reproducible local checks. It exercises temporal recall, updated versus archived memories, explicit forgetting, low-confidence abstention, sensitive-memory boundaries, and prompt-injection handling across the checked-in strategies (`raw-history`, `semantic-rag`, and `alfredo`). It is useful for inspecting per-question decisions, evidence, ignored IDs, and context accounting—not for making a universal quality claim.

This benchmark is **not a security or privacy audit**, is not production data, and does not establish that a deployment is safe for secrets or regulated workloads. Results depend on the fixture, configuration, and offline run; validate your own data, policies, and threat model.

To run the documented offline comparison from a checkout:

```bash
python -m memory_agent --offline benchmark compare \
  --users benchmarks/alfredos_vault/users.json \
  --memories benchmarks/alfredos_vault/memories.jsonl \
  --questions benchmarks/alfredos_vault/evaluation_questions.jsonl \
  --report .alfredo/benchmark-comparison.json \
  --seed 42 --run local-offline
```

## Integration and community

- [Lifecycle demo](examples/demo_lifecycle.py) — a deterministic four-stage adoption path.
- [MCP and Python integration](INTEGRATION.md) — stdio/HTTP server setup, namespaces, evidence, and providers.
- [Benchmark fixtures and reports](benchmarks/alfredos_vault/) — synthetic inputs and generated outputs.
- [License](LICENSE) — MIT terms for this repository.
- [Security policy](SECURITY.md) — security reporting and deployment guidance.
- [Contributing](CONTRIBUTING.md) — development and review workflow.
- [Roadmap](ROADMAP.md) — planned work and scope boundaries.

The community-policy links are kept in the README so adopters have stable navigation as those documents evolve.

## License

Alfredo is released under the [MIT License](LICENSE).
