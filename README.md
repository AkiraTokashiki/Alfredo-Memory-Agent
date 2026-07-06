# Alfredo — MemoryAgent

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
| **43 tests** | Full coverage of storage, forgetting curve, retrieval, and agent loop |

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
4. keeps critical memories inside a bounded recall context even after low-importance noise is added;
5. prints active/archived memory stats for judging.

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
│   └── commands.py            # chat, stats, search, mcp, llm
├── integrations/              # Extensions
│   ├── mcp_server.py          # MCP server (stdio + HTTP)
│   └── llm_connector.py       # DeepSeek/OpenAI/Anthropic connector
└── __main__.py                # Entry point
```

---

## 🧪 Test Suite

```
43 passed in ~25s

tests/test_memory_store.py  — 14 tests (CRUD, embeddings, tags, sessions)
tests/test_forgetting.py    — 12 tests (curve, reinforce, archival, lifespan)
tests/test_retrieval.py     —  8 tests (scoring, MMR, filters, access tracking)
tests/test_agent.py         —  9 tests (full cycle, multi-session, stats)
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

See [INTEGRACION.md](./INTEGRACION.md) for full details.

---

## 🏆 Hackathon Submission

- **Track**: Track 1 — MemoryAgent.
- **Architecture diagram**: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).
- **Submission checklist and description**: [`SUBMISSION.md`](./SUBMISSION.md).
- **Alibaba Cloud proof code**: [`deploy/alibaba_cloud_proof.py`](./deploy/alibaba_cloud_proof.py).
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

---

## 🛠 Tech Stack

- **Python 3.11+**
- **SQLite** (WAL mode, persistent)
- **sentence-transformers** (all-MiniLM-L6-v2, 384-dim embeddings)
- **NumPy** (cosine similarity, MMR)
- **MCP SDK** (Model Context Protocol)
- **Click** (CLI)
- **Pytest** (43 tests)

---

## 📄 License

MIT — use it, fork it, hack it.

---

## 🏆 Hackathon Tips

- **Demo in 30 seconds**: `python examples/demo_basic.py`
- **Show persistence**: run `demo_multi_session.py` to prove cross-session recall
- **Show forgetting**: importance 0.3 memories decay in ~3 days
- **Show retrieval quality**: "I use Linux" vs "prefer Windows" — the semantic search finds the right one
- **MCP integration demo**: connect to Hermes, ask about a preference from a previous session
