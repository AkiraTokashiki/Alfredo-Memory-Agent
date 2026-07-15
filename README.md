# Memo — MemoryAgent

![Memo MemoryAgent lifecycle: learn, retrieve, trust, pack, reinforce, supersede or forget](docs/assets/Memo-memory-lifecycle.svg)

> **A local MemoryAgent that learns, remembers, forgets, and explains what it recalls.**
> Memo keeps a durable, selective memory for an agent without turning your data into a hosted SaaS.

**Start here:** [offline quickstart](#quickstart) · [lifecycle demo](examples/demo_lifecycle.py) · [MCP and Python integration](INTEGRATION.md) · [synthetic benchmark](#benchmark-evidence)

---

## What Memo is

Memo is a Python SDK and CLI for a local memory layer. Its native store is SQLite, and the intended distribution name is `Memo-memory-agent` while the import namespace remains `memory_agent`. You run it beside your agent and choose where its vault lives; there is no Memo-hosted dashboard, tenant service, billing account, or required remote memory API.

A conversation transcript is not a memory policy. Raw history grows without a bounded selection step, while a simple RAG index can return semantically similar text without deciding whether it is trusted, stale, superseded, or safe to pack into a prompt. Memo makes those decisions explicit and inspectable:

- **Namespaces** keep records and sessions scoped to an agent or tenant boundary.
- **Evidence** records score, matching signals, trust, and a reason for each retrieval decision.
- **Selected and dropped IDs** show what entered the bounded recall packet and what did not.
- **Lifecycle state** makes reinforcement, supersession, archival, and explicit forgetting visible.

These are SDK behaviors, not a promise of automatic compliance or production security. Configure storage, access control, retention, and deployment isolation for your environment.

## Quickstart

The canonical install contract uses the `Memo-memory-agent` distribution name and the `memory_agent` module:

```bash
pip install Memo-memory-agent
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

Conceptually, Memo's lifecycle can be summarized by this bounded path; it is not the implementation order of every `perceive` turn:

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

The demo covers cross-session recall, preference supersession, and bounded trusted context without network access, model downloads, API keys, or wall-clock output. See [the architecture lifecycle](docs/ARCHITECTURE.md#real-lifecycle) for the actual perceive-turn stages and [the full integration guide](INTEGRATION.md) for MCP stdio/HTTP and programmatic usage.

## Why it is different

| Approach | What it keeps | What it does not decide | What Memo adds |
| --- | --- | --- | --- |
| Raw conversation history | The transcript | Which facts are current, trusted, or within budget | Candidate extraction, lifecycle state, bounded packets, and explicit forget/supersede operations |
| Simple semantic RAG | Indexed chunks ranked by similarity | Whether a result is low-confidence, stale, superseded, or safe for context | Trust evidence, namespace-aware retrieval, selected/dropped IDs, and reinforcement/decay |
| Memo MemoryAgent | Structured local memories in a SQLite vault | Your deployment's access-control and privacy obligations | A local, inspectable learn → retrieve → trust → pack → reinforce → supersede/forget lifecycle |

The comparison is about responsibilities, not a claim that one strategy wins every workload. Memo can be embedded through the protocols in the SDK and can also expose an MCP server; it is not a hosted memory SaaS.

## Agentic memory primitives

Memo now exposes the higher-level primitives needed to build memory workflows without replacing the SQLite core:

- **Typed relations** — `MemoryRelation` edges such as `supports`, `supersedes`, and `contradicts` are namespace-aware, lifecycle-aware, auditable, and safe to expand only after trust and context-budget checks.
- **Proposal-first evolution** — `EvolutionProposal` and `EvolutionDecision` let a deterministic or remote planner suggest metadata changes, supersession, and relations. `MemoryStore.apply_evolution()` validates the proposal and commits accepted mutations plus one audit event atomically; rejected proposals are recorded without storing raw prompts or secrets.
- **Procedural task packs** — `TaskMemoryPackStore` persists task-specific triggers, instructions, constraints, required memory IDs, and examples using `memory_type="procedural"`. `build_task_context()` keeps task recall namespace-scoped and bounded.
- **Episodic consolidation** — `EpisodeSummaryBuilder` creates deterministic summaries from session events, while `consolidate_session()` uses an idempotency key so reopening a store cannot duplicate a session summary.

The public SDK exports these building blocks from `memory_agent`, including `MemoryRelation`, `EvolutionProposal`, `EvolutionDecision`, `TaskMemoryPack`, `TaskMemoryPackStore`, `EpisodeSummary`, and `build_task_context`. They remain local SQLite operations; planner output is never a direct state mutation.

## Inspectable Markdown export

SQLite remains the source of truth, but active memories can be projected into deterministic, human-readable Markdown files without adding an Obsidian runtime dependency:

```bash
Memo --db .Memo/memory.db export-markdown \
  --namespace tenant-a \
  --output .Memo/markdown/tenant-a
```

The exporter writes one Memo-owned `<memory_id>.md` file per active memory in the requested namespace, with stable frontmatter for identity, type, confidence, lifecycle, namespace, and update time. Export is read-only, repeatable, and never crosses namespace boundaries. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the clean-room and attribution boundary.


## Architecture in brief

```text
agent input
    │
    ▼
MemoryAgent orchestrator ──► extraction/consolidation ──► namespace-scoped SQLite vault
    │                                  │                           │
    │                                  └─ reinforce/supersede/forget┘
    ▼
retrieval ──► typed relations + trust policy + evidence ──► bounded recall packet
    │                                                        (selected_ids, dropped_ids)
    ▼
agent context / response
    │
    └─► proposal-first evolution / procedural task packs / episodic consolidation
```

The offline embedding engine is deterministic. The optional semantic provider is selected explicitly. Storage, embedding, retrieval, and trust boundaries are injectable through the SDK ports, while `INTEGRATION.md` documents the CLI, MCP, and Python entry points.

## Benchmark evidence

The [Memo's Vault fixtures](benchmarks/Memos_vault/) are a **synthetic** comparison dataset for reproducible local checks. It exercises temporal recall, updated versus archived memories, explicit forgetting, low-confidence abstention, sensitive-memory boundaries, and prompt-injection handling across the checked-in strategies (`raw-history`, `semantic-rag`, and `Memo`). It is useful for inspecting per-question decisions, evidence, ignored IDs, and context accounting—not for making a universal quality claim.

This benchmark is **not a security or privacy audit**, is not production data, and does not establish that a deployment is safe for secrets or regulated workloads. Results depend on the fixture, configuration, and offline run; validate your own data, policies, and threat model.

The comparison also has an opt-in `Memo-agentic` strategy (`config={"agentic": true}`). It is a deterministic metadata view over the same local baseline behavior: structured memories are joined to typed relations, proposal-first evolution decisions, task packs, episodic consolidation/deduplication, forgetting and trust decisions, and bounded context accounting. Agentic rows expose selected and dropped IDs, trust evidence, relation IDs/types, evolution and audit IDs, task-pack IDs, episode deduplication, context size, and latency. The default comparison remains exactly the three baseline strategies; no remote model or API is used.

Agentic evidence is fixture-derived and marked synthetic; dataset hashes and the run seed make repeated offline runs comparable. These fields describe exercised decisions, not a production guarantee. The benchmark cannot establish privacy, security, authorization, retention, deletion, or quality for a deployment, and it must not be read as an endorsement of any external memory framework. The terminology is informed by public discussions such as [MemGPT/Letta](https://github.com/cpacker/MemGPT) and [Graphiti](https://github.com/getzep/graphiti) for provenance only; Memo does not reuse their code or imply affiliation.

To run the documented offline comparison from a checkout:

```bash
python -m memory_agent --offline benchmark compare --users benchmarks/Memos_vault/users.json --memories benchmarks/Memos_vault/memories.jsonl --questions benchmarks/Memos_vault/evaluation_questions.jsonl --report .Memo/benchmark-comparison.json --seed 42 --run local-offline
```

## Integration and community

- [Lifecycle demo](examples/demo_lifecycle.py) — a deterministic four-stage adoption path.
- [MCP and Python integration](INTEGRATION.md) — stdio/HTTP server setup, namespaces, evidence, and providers.
- [Benchmark fixtures and reports](benchmarks/Memos_vault/) — synthetic inputs and generated outputs.
- [Provenance and export boundary](docs/PROVENANCE.md) — Markdown projection, licenses, and clean-room notes.
- [License](LICENSE) — MIT terms for this repository.
- [Security policy](SECURITY.md) — security reporting and deployment guidance.
- [Contributing](CONTRIBUTING.md) — development and review workflow.
- [Roadmap](ROADMAP.md) — planned work and scope boundaries.

The community-policy links are kept in the README so adopters have stable navigation as those documents evolve.

## License

Memo is released under the [MIT License](LICENSE).
