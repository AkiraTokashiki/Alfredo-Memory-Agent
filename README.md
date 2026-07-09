# Alfredo — MemoryAgent
## [Video Demo Link](https://youtu.be/bg0th6M7qec)
**Give your Qwen Cloud agent a brain that remembers, forgets, and learns.**

Alfredo is a MemoryAgent that gives Qwen Cloud agents persistent, selective memory across sessions, recalling key user preferences and facts while filtering context with relevance, recency, importance, and decay.

---

## ✨ Features

| Feature | What it does |
|---------|-------------|
| **Semantic memory** | sentence-transformers embeddings (384d) find relevant memories by meaning, not keywords |
| **Ebbinghaus forgetting** | Memories decay exponentially — high-importance lasts ~90 days, low-importance fades in ~3 days |
| **Reinforcement** | Each retrieval boosts recall strength (+0.15), keeping useful memories alive |
| **MMR diversity** | Maximum Marginal Relevance prevents near-duplicate results |
| **Multi-factor scoring** | 40% semantic + 20% recency + 20% importance + 20% strength |
| **Automatic archival** | Memories below strength threshold are archived, not deleted |
| **Auto-extraction** | Detects preferences ("I like X"), habits, and facts from natural conversation |
| **MCP native** | Drop-in integration with Hermes, Claude Desktop, Cursor — any MCP client |
| **Multi-session** | Memories persist across sessions, survive restarts |
| **Vault benchmark** | Synthetic sustained-memory benchmark: 25 users, 5,000 memories, 500 evaluation questions |
| **Trust decisions** | Filters expired, forgotten, superseded, low-confidence, and prompt-injection memories before prompt injection |
| **Automated tests** | Coverage for storage, forgetting curve, retrieval, agent loop, video demo, and the vault benchmark |

---

## 🔬 The Science

### Forgetting Curve (Ebbinghaus)

```
strength(t) = initial_strength × e^(-t / decay_constant)
```

| Importance | Decay constant | Lifespan before archival |
|-----------|---------------|------------------------|
| High (≥ 0.8) | 720 hours | ~90 days |
| Medium (≥ 0.5) | 168 hours | ~21 days |
| Low (< 0.5) | 24 hours | ~3 days |

Each retrieval reinforces: `strength = min(1.0, strength + 0.15)`

### Retrieval Scoring

```
total_score = 0.40 × semantic_similarity
            + 0.20 × recency
            + 0.20 × importance
            + 0.20 × recall_strength
```

MMR diversity penalty then ensures you don't get 3 nearly-identical results.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/AkiraTokashiki/Alfredo.git
cd Alfredo
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -e .
pip install -e ".[all]"     # with MCP + LLM support
```

### Native memory vault

By default, Alfredo stores runtime memory in a native SQLite vault instead of
the current working directory.

Development checkout default:

```text
.alfredo/memory_agent.db
```

Override with:

```bash
set ALFREDO_HOME=E:\code\alfredo\.alfredo   # Windows CMD
export ALFREDO_HOME="$PWD/.alfredo"          # Linux/macOS
```

You can still pass an explicit DB path:

```bash
python -m memory_agent --db path/to/memory_agent.db chat
```

### 2. Run the demo

```bash
python examples/demo_basic.py
```

### Hackathon demo

```bash
python examples/demo_hackathon.py
```

This demo shows the complete memory lifecycle:

1. learns a user preference in one session;
2. recalls it in a later session;
3. archives the stale preference when the user changes it;
4. explains that the SQLite vault can be large while each prompt receives only a small recall packet;
5. runs the synthetic Alfredo's Vault benchmark for judge-friendly proof.

### No-voice Devpost video demo

For screen recording with large terminal captions:

```bash
python examples/demo_video.py
```

Open PowerShell or CMD full screen, increase the font size, run the command,
and record the terminal output. The script uses large English captions and
pauses between scenes, so it works without voiceover. The recording is designed
to stay around 1.5-2 minutes, safely under the 3-minute Devpost limit.

### Alfredo's Vault benchmark

The benchmark uses only synthetic data: 25 fictional users, 5,000 JSONL
memories, and 500 evaluation questions covering temporal recall,
contradiction updates, expiry, explicit forgetting, low-confidence
abstention, sensitive-memory boundaries, and prompt-injection resistance.

Seed the benchmark into SQLite:

```bash
python -m memory_agent --db .alfredo/vault_benchmark.db benchmark seed \
  --users benchmarks/alfredos_vault/users.json \
  --memories benchmarks/alfredos_vault/memories.jsonl \
  --expected-users 25 \
  --expected-memories 5000
```

Evaluate the seeded vault:

```bash
python -m memory_agent --db .alfredo/vault_benchmark.db benchmark run \
  --users benchmarks/alfredos_vault/users.json \
  --questions benchmarks/alfredos_vault/evaluation_questions.jsonl \
  --report benchmarks/alfredos_vault/benchmark_report.json \
  --expected-users 25 \
  --expected-questions 500
```

The report records per-question answers, retrieved memory IDs, ignored memory
IDs, confidence, behavior detected, pass/fail status, and security events.

### 3. Interactive chat

```bash
python -m memory_agent chat
```

```
  You > I like programming in Python
  [+] preference: The user prefers: programming in python

  You > What language do I like?
  ─────────────────────────────────────────
  [Retrieved memories]:
    1. [preference] (importance=0.7, strength=1.00) The user prefers: programming in python
  ─────────────────────────────────────────
  I remembered that the user prefers: programming in python...
```

### 4. Integrate with Hermes (MCP)

```bash
hermes mcp add memory-agent \
  --command python \
  --args "-m,memory_agent,mcp"
```

Then in any Hermes session, use `/reload-mcp` and call `memory__perceive`, `memory__search`, `memory__store`, etc.

### 5. Connect to Qwen Cloud

```bash
export DASHSCOPE_API_KEY=sk-...
python -m memory_agent llm --provider qwencloud
```

Full conversation with a Qwen Cloud model that **remembers everything** between turns and sessions.

### 6. Connect to DeepSeek / OpenAI

```bash
export DEEPSEEK_API_KEY=sk-...
python -m memory_agent llm --provider deepseek
```

The same memory layer works with any configured OpenAI-compatible provider.

---

## 🏗 Architecture

```
                    User Input
                        │
                        ▼
┌──────────────────────────────────────┐
│          MemoryAgent                  │
│                                       │
│  1. PERCEIVE ──► 2. EXTRACT ──► 3. RETRIEVE
│                     │            │      │
│               preferences      MMR    scoring
│               + facts          │      │
│                     ▼          ▼      ▼
│              New Memories   Recollections
│                                       │
│  4. FORMAT CONTEXT ──► 5. DECAY ──► 6. ARCHIVE
└──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────┐
│          SQLite Store                 │
│  ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │Memories  │ │Embeddings│ │Tags   │ │
│  └──────────┘ └──────────┘ └───────┘ │
└──────────────────────────────────────┘
```

### Module map

```
src/memory_agent/
├── core/                      # Engine room
│   ├── memory_store.py        # SQLite persistence (WAL mode, 5 tables)
│   ├── embeddings.py          # sentence-transformers (all-MiniLM-L6-v2)
│   ├── forgetting.py          # Ebbinghaus curve + reinforcement
│   ├── retrieval.py           # Scoring + MMR diversity
│   └── config.py              # All tunable parameters
├── agent/                     # Agent loop
│   ├── orchestrator.py        # Perceive → extract → retrieve → decay
│   └── decision.py            # NLP extraction of preferences/facts
├── cli/                       # Interactive CLI (Click)
│   └── commands.py            # chat, stats, search, benchmark, mcp, llm
├── integrations/              # Extensions
│   ├── mcp_server.py          # MCP server (stdio + HTTP)
│   └── llm_connector.py       # DeepSeek/OpenAI/Anthropic connector
├── benchmark.py               # Synthetic vault benchmark loader/evaluator
└── __main__.py                # Entry point
```

---

## 🧪 Test Suite

```
tests/test_memory_store.py   — SQLite CRUD, embeddings, tags, sessions
tests/test_forgetting.py     — forgetting curve, reinforcement, archival
tests/test_retrieval.py      — scoring, MMR, filters, access tracking
tests/test_agent.py          — full memory cycle, multi-session recall, stats
tests/test_benchmark.py      — vault benchmark loading, seeding, trust-policy evaluation
tests/test_video_demo.py     — no-voice Devpost benchmark demo smoke tests
```

```bash
pytest tests/ -v
```

---

## 🔌 Integrations

| Integration | How |
|------------|-----|
| **Qwen Cloud** | `DASHSCOPE_API_KEY=... python -m memory_agent llm --provider qwencloud` |
| **Hermes Agent** | `hermes mcp add memory-agent` → 6 tools auto-register |
| **Claude Desktop** | Add MCP server in claude_desktop_config.json |
| **Cursor** | Add to `.cursor/mcp.json` |
| **Any MCP client** | Stdio: `python -m memory_agent mcp` or HTTP: `python -m memory_agent mcp --http --port 8090` |
| **DeepSeek / OpenAI / Anthropic** | `python -m memory_agent llm --provider deepseek` |
| **Custom Python** | `from memory_agent import MemoryAgent` |

See [INTEGRATION.md](./INTEGRATION.md) for full details.

---

## 🏆 Hackathon Submission

- **Track**: Track 1 — MemoryAgent.
- **Architecture diagram**: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).
- **Submission checklist and description**: [`SUBMISSION.md`](./SUBMISSION.md).
- **Alibaba Cloud proof code**: [`deploy/alibaba_cloud_proof.py`](./deploy/alibaba_cloud_proof.py).
- **Vault benchmark**: [`benchmarks/alfredos_vault/`](./benchmarks/alfredos_vault/) contains synthetic users, 5,000 memories, 500 evaluation questions, and generated reports.
- **License**: [`LICENSE`](./LICENSE), MIT.

Before submitting to Devpost, publish this repository publicly, upload the demo video to YouTube/Vimeo/Facebook Video, and add the final URLs to `SUBMISSION.md`.

---

## 📊 Stats & Commands

In the interactive CLI:

```
/stats      — View memory statistics (type distribution, decay lifespans)
/search <q> — Semantic search
/memories   — List all active memories
/forget <id>— Delete a memory
/help       — Show commands
/quit       — Exit
```

Benchmark CLI:

```bash
python -m memory_agent --db .alfredo/vault_benchmark.db benchmark seed --users benchmarks/alfredos_vault/users.json --memories benchmarks/alfredos_vault/memories.jsonl
python -m memory_agent --db .alfredo/vault_benchmark.db benchmark run --users benchmarks/alfredos_vault/users.json --questions benchmarks/alfredos_vault/evaluation_questions.jsonl --report benchmarks/alfredos_vault/benchmark_report.json
```

---

## 🛠 Tech Stack

- **Python 3.11+**
- **SQLite** (WAL mode, persistent)
- **sentence-transformers** (all-MiniLM-L6-v2, 384-dim embeddings)
- **NumPy** (cosine similarity, MMR)
- **MCP SDK** (Model Context Protocol)
- **Click** (CLI)
- **Pytest**

---

## 📄 License

MIT — use it, fork it, hack it.

---

## 🏆 Tips

- **Demo in 30 seconds**: `python examples/demo_basic.py`
- **Show benchmarked scale**: run `python examples/demo_video.py` to show 25 users, 5,000 memories, 500 questions, and trust-policy decisions
- **Show persistence**: run `demo_multi_session.py` to prove cross-session recall
- **Show forgetting**: use the benchmark expiry/forgotten-memory cases
- **Show retrieval quality**: show source, date, confidence, supersession, and ignored-memory reasons
- **MCP integration demo**: connect to Hermes, ask about a preference from a previous session
