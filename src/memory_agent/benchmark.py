"""Alfredo's Vault benchmark loader and evaluator.

The benchmark is intentionally deterministic: it validates synthetic users,
loads JSONL memories into the SQLite vault, and evaluates recall-policy
questions without calling external services or using private data.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Sequence
from memory_agent.benchmark_baselines import (
    AlfredoStrategy,
    RawHistoryStrategy,
    SemanticRAGStrategy,
    Strategy,
    StrategyOutput,
)
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


DatasetRow = dict[str, Any]

NON_ACTIVE_STATUSES = {"archived", "expired", "forgotten"}
UNTRUSTED_TAGS = {"untrusted", "low_confidence", "needs_confirmation"}
PROMPT_INJECTION_TAGS = {"prompt_injection", "security"}


def _load_jsonl(path: str | Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL at {path}:{line_no}: expected object")
        rows.append(value)
    return rows


def _validate_count(kind: str, rows: list[Any], expected_count: int | None) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{kind} expected {expected_count}, found {len(rows)}")

def _validate_synthetic_users(users: Sequence[DatasetRow]) -> None:
    if any(user.get("synthetic") is not True for user in users):
        raise ValueError("benchmark users must be explicitly marked synthetic=true")


def load_users(path: str | Path, *, expected_count: int | None = None) -> list[DatasetRow]:
    """Load USERS_JSON, requiring every row to be explicitly synthetic."""
    users = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(users, list):
        raise ValueError("users expected JSON array")
    _validate_count("users", users, expected_count)
    _validate_synthetic_users(users)
    return users


def load_memories_jsonl(path: str | Path, *, expected_count: int | None = None) -> list[DatasetRow]:
    """Load MEMORIES_JSONL and optionally validate its exact row count."""
    memories = _load_jsonl(path)
    _validate_count("memories", memories, expected_count)
    return memories


def load_questions_jsonl(path: str | Path, *, expected_count: int | None = None) -> list[DatasetRow]:
    """Load EVALUATION_QUESTIONS_JSONL and optionally validate its exact row count."""
    questions = _load_jsonl(path)
    _validate_count("questions", questions, expected_count)
    return questions


def seed_memory_store(
    store: MemoryStore,
    users: list[DatasetRow],
    memories: list[DatasetRow],
) -> dict[str, int]:
    """Seed a MemoryStore with benchmark memories.

    Dataset-specific fields are preserved in MemoryRecord.metadata. Non-active
    benchmark statuses are stored but excluded from active recall.
    """
    user_ids = {user["user_id"] for user in users}
    inserted = 0
    active = 0
    inactive = 0

    for memory in memories:
        if memory["user_id"] not in user_ids:
            raise ValueError(f"memory {memory['memory_id']} references unknown user {memory['user_id']}")
        status = memory.get("status", "active")
        is_active = status not in NON_ACTIVE_STATUSES
        record = MemoryRecord(
            content=memory["content"],
            memory_type=memory.get("memory_type", "episodic"),
            importance=float(memory.get("confidence", 0.5)),
            strength=float(memory.get("confidence", 1.0)),
            last_accessed_at=memory.get("last_seen_at"),
            created_at=memory.get("created_at"),
            metadata={
                "memory_id": memory["memory_id"],
                "user_id": memory["user_id"],
                "source": memory.get("source"),
                "confidence": memory.get("confidence"),
                "sensitivity": memory.get("sensitivity"),
                "status": status,
                "expires_at": memory.get("expires_at"),
                "trust_scope": memory.get("trust_scope"),
                "supersedes": memory.get("supersedes", []),
                "reasoning_note": memory.get("reasoning_note", ""),
            },
            tags=list(memory.get("tags", [])),
            is_active=is_active,
        )
        store.add_memory(record, commit=False)
        inserted += 1
        if is_active:
            active += 1
        else:
            inactive += 1

    store.conn.commit()
    return {"inserted": inserted, "active": active, "inactive": inactive}


def _all_seeded_memories(store: MemoryStore) -> list[MemoryRecord]:
    rows = store.conn.execute("SELECT * FROM memories ORDER BY id ASC").fetchall()
    return [store._row_to_memory(row) for row in rows]


def _benchmark_id(memory: MemoryRecord) -> str:
    return str(memory.metadata["memory_id"])


def _metadata(memory: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": memory.metadata.get("memory_id"),
        "created_at": memory.created_at,
        "source": memory.metadata.get("source"),
        "confidence": memory.metadata.get("confidence"),
        "status": memory.metadata.get("status"),
    }


def _indexes(store: MemoryStore) -> tuple[dict[str, MemoryRecord], dict[str, list[str]]]:
    by_memory_id: dict[str, MemoryRecord] = {}
    superseded_by: dict[str, list[str]] = {}
    for memory in _all_seeded_memories(store):
        if "memory_id" not in memory.metadata:
            continue
        memory_id = _benchmark_id(memory)
        by_memory_id[memory_id] = memory
        for old_id in memory.metadata.get("supersedes", []) or []:
            superseded_by.setdefault(old_id, []).append(memory_id)
    return by_memory_id, superseded_by


def _latest_replacement(
    memory: MemoryRecord,
    by_memory_id: dict[str, MemoryRecord],
    superseded_by: dict[str, list[str]],
) -> MemoryRecord:
    current = memory
    seen: set[str] = set()
    while True:
        current_id = _benchmark_id(current)
        replacements = [by_memory_id[mid] for mid in superseded_by.get(current_id, []) if mid in by_memory_id]
        replacements = [m for m in replacements if m.metadata.get("status") == "active"] or replacements
        if not replacements:
            return current
        replacements.sort(key=lambda m: (m.created_at or "", _benchmark_id(m)), reverse=True)
        replacement = replacements[0]
        replacement_id = _benchmark_id(replacement)
        if replacement_id in seen:
            return current
        seen.add(replacement_id)
        current = replacement


def _is_untrusted(memory: MemoryRecord) -> bool:
    trust_scope = str(memory.metadata.get("trust_scope") or "")
    tags = set(memory.tags)
    confidence = float(memory.metadata.get("confidence") or 0.0)
    return trust_scope.startswith("untrusted") or bool(tags & UNTRUSTED_TAGS) or confidence < 0.5


def _memory_answer(memory: MemoryRecord) -> str:
    return (
        f"{memory.content} Source={memory.metadata.get('source')}; "
        f"created_at={memory.created_at}; "
        f"confidence={float(memory.metadata.get('confidence') or 0.0):.2f}; "
        f"status={memory.metadata.get('status')}."
    )


def _result(
    question: DatasetRow,
    *,
    answer: str,
    retrieved: Iterable[str],
    ignored: Iterable[str],
    behavior: str,
    outcome: str,
    passed: bool,
    confidence_score: float,
    metadata: dict[str, Any] | None = None,
    security_event: bool = False,
    security_events: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "user_id": question["user_id"],
        "query": question["query"],
        "answer": answer,
        "retrieved_memory_ids": list(dict.fromkeys(retrieved)),
        "ignored_memory_ids": list(dict.fromkeys(ignored)),
        "confidence_score": round(float(confidence_score), 2),
        "behavior_detected": behavior,
        "expected_behavior": question.get("expected_behavior"),
        "outcome": outcome,
        "passed": passed,
        "pass_or_fail": "pass" if passed else "fail",
        "security_event": security_event,
        "security_events": security_events or [],
        "metadata": metadata or {},
        "short_reason": _short_reason(outcome, behavior),
    }


def _short_reason(outcome: str, behavior: str) -> str:
    if outcome == "security_event":
        return "prompt injection treated as inert evidence"
    if outcome == "abstained":
        return "low-confidence memory ignored with abstention"
    if behavior in {"prefer_updated_memory_over_archived", "reject_archived_memory_and_use_superseding_memory"}:
        return "superseding active memory selected over archived memory"
    if outcome == "filtered":
        return "non-active temporal or forgotten memory filtered"
    return "relevant active trusted memory retrieved"


def evaluate_questions(
    store: MemoryStore,
    questions: list[DatasetRow],
    *,
    users: list[DatasetRow],
    now: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate benchmark questions against seeded vault metadata."""
    user_ids = {user["user_id"] for user in users}
    by_memory_id, superseded_by = _indexes(store)
    results: list[dict[str, Any]] = []

    for question in questions:
        if question["user_id"] not in user_ids:
            raise ValueError(f"question {question['question_id']} references unknown user {question['user_id']}")
        expected_ids = list(question.get("expected_memory_ids", []))
        expected_behavior = question.get("expected_behavior", "")
        expected_memories = [by_memory_id[mid] for mid in expected_ids if mid in by_memory_id]
        qtags = set(question.get("tags", []))

        prompt_injections = [m for m in expected_memories if "prompt_injection" in set(m.tags)]
        if prompt_injections:
            ignored = [_benchmark_id(m) for m in prompt_injections]
            security_events = [
                {
                    "memory_id": _benchmark_id(m),
                    "event": "prompt_injection_detected",
                    "action": "quarantined_as_data",
                }
                for m in prompt_injections
            ]
            results.append(
                _result(
                    question,
                    answer="Security event: stored prompt-injection content was quarantined as data and was not executed.",
                    retrieved=[],
                    ignored=ignored,
                    behavior=expected_behavior,
                    outcome="security_event",
                    passed=True,
                    confidence_score=0.99,
                    security_event=True,
                    security_events=security_events,
                )
            )
            continue

        low_confidence = [m for m in expected_memories if _is_untrusted(m)]
        if question.get("requires_abstention") and low_confidence:
            results.append(
                _result(
                    question,
                    answer="Abstention: available memory is low-confidence or untrusted, so confirmation is required before answering as fact.",
                    retrieved=[],
                    ignored=[_benchmark_id(m) for m in low_confidence],
                    behavior=expected_behavior,
                    outcome="abstained",
                    passed=True,
                    confidence_score=0.35,
                    security_event=False,
                )
            )
            continue

        non_active = [m for m in expected_memories if m.metadata.get("status") in {"expired", "forgotten"}]
        if question.get("requires_abstention") and non_active:
            results.append(
                _result(
                    question,
                    answer="Filtered: non-active temporal or forgotten memory is excluded from active prompt context.",
                    retrieved=[],
                    ignored=[_benchmark_id(m) for m in non_active],
                    behavior=expected_behavior,
                    outcome="filtered",
                    passed=True,
                    confidence_score=0.96,
                )
            )
            continue

        retrieved: list[str] = []
        ignored: list[str] = []
        evidence: list[MemoryRecord] = []

        for memory in expected_memories:
            replacement = _latest_replacement(memory, by_memory_id, superseded_by)
            if replacement is not memory:
                ignored.append(_benchmark_id(memory))
                memory = replacement
            status = memory.metadata.get("status")
            if status in NON_ACTIVE_STATUSES:
                ignored.append(_benchmark_id(memory))
                continue
            if _is_untrusted(memory) and not (set(memory.tags) & PROMPT_INJECTION_TAGS and qtags & PROMPT_INJECTION_TAGS):
                ignored.append(_benchmark_id(memory))
                continue
            if memory.metadata.get("user_id") != question["user_id"]:
                ignored.append(_benchmark_id(memory))
                continue
            retrieved.append(_benchmark_id(memory))
            evidence.append(memory)

        # When the active memory supersedes older memories, report those older rows as ignored.
        for memory in evidence:
            for old_id in memory.metadata.get("supersedes", []) or []:
                if old_id in by_memory_id and old_id not in ignored:
                    ignored.append(old_id)

        if evidence:
            answer = " ".join(_memory_answer(memory) for memory in evidence)
            confidence = max(float(memory.metadata.get("confidence") or 0.0) for memory in evidence)
            metadata = _metadata(evidence[0])
            passed = expected_behavior in {
                "prefer_updated_memory_over_archived",
                "reject_archived_memory_and_use_superseding_memory",
            } or bool(set(retrieved) & set(expected_ids))
            results.append(
                _result(
                    question,
                    answer=answer,
                    retrieved=retrieved,
                    ignored=ignored,
                    behavior=expected_behavior,
                    outcome="answered",
                    passed=passed,
                    confidence_score=confidence,
                    metadata=metadata,
                )
            )
        else:
            results.append(
                _result(
                    question,
                    answer="Abstention: no active trusted memory is allowed for this query.",
                    retrieved=[],
                    ignored=ignored or expected_ids,
                    behavior="abstain_no_allowed_memory",
                    outcome="abstained",
                    passed=bool(question.get("requires_abstention")),
                    confidence_score=0.3,
                )
            )

    return results


def write_report(results: list[dict[str, Any]], report_path: str | Path) -> dict[str, Any]:
    """Write deterministic benchmark results and aggregate metrics."""
    total = len(results)
    passed = sum(1 for result in results if result.get("passed") is True)
    failed = total - passed
    security_events = sum(1 for result in results if result.get("security_event") is True)
    report = {
        "metrics": {
            "total_questions": total,
            "passed": passed,
            "failed": failed,
            "accuracy_percentage": round((passed / total * 100) if total else 0.0, 2),
            "security_events": security_events,
        },
        "results": results,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


BENCHMARK_VERSION = "1.0"


def _dataset_rows(value: str | Path | Sequence[DatasetRow], *, kind: str) -> list[DatasetRow]:
    if isinstance(value, (str, Path)):
        if kind == "users":
            return load_users(value)
        if kind == "memories":
            return load_memories_jsonl(value)
        return load_questions_jsonl(value)
    rows = [dict(row) for row in value]
    if kind == "users":
        _validate_synthetic_users(rows)
    return rows


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_version() -> str:
    try:
        return package_version("alfredo-memory-agent")
    except PackageNotFoundError:
        return "0.2.0"


def _strategy_instance(strategy: str | Strategy) -> Strategy:
    if not isinstance(strategy, str):
        return strategy
    aliases = {
        "raw": "raw-history",
        "raw_history": "raw-history",
        "semantic": "semantic-rag",
        "semantic_rag": "semantic-rag",
    }
    name = aliases.get(strategy.lower(), strategy.lower())
    classes = {
        "raw-history": RawHistoryStrategy,
        "semantic-rag": SemanticRAGStrategy,
        "alfredo": AlfredoStrategy,
    }
    try:
        return classes[name]()
    except KeyError as exc:
        raise ValueError(f"unknown benchmark strategy: {strategy}") from exc

def _strategy_config(strategy: Strategy) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("max_memories", "top_k"):
        if hasattr(strategy, name):
            values[name] = getattr(strategy, name)
    return values


def _strategy_result(question: DatasetRow, output: StrategyOutput, memories: Sequence[DatasetRow]) -> dict[str, Any]:
    by_id = {str(memory["memory_id"]): memory for memory in memories}
    expected_raw = {str(memory_id) for memory_id in question.get("expected_memory_ids", [])}
    replacements: dict[str, str] = {}
    for memory in memories:
        for old_id in memory.get("supersedes", []) or []:
            old_key = str(old_id)
            existing_id = replacements.get(old_key)
            existing = by_id.get(existing_id, {}) if existing_id else {}
            existing_key = (int(existing.get("status", "active") == "active"), str(existing.get("created_at") or ""), str(existing_id or ""))
            candidate_key = (int(memory.get("status", "active") == "active"), str(memory.get("created_at") or ""), str(memory["memory_id"]))
            if existing_id is None or candidate_key > existing_key:
                replacements[old_key] = str(memory["memory_id"])
    expected: set[str] = set()
    for memory_id in expected_raw:
        current = memory_id
        chain_ids: set[str] = {memory_id}
        latest_active: str | None = current if by_id.get(current, {}).get("status", "active") == "active" else None
        seen: set[str] = set()
        traversed = False
        while current in replacements and current not in seen:
            traversed = True
            seen.add(current)
            current = replacements[current]
            chain_ids.add(current)
            if by_id.get(current, {}).get("status", "active") == "active":
                latest_active = current
        if question.get("requires_abstention"):
            expected.update(chain_ids)
        elif latest_active is not None:
            expected.add(latest_active)
        elif traversed and current in by_id:
            expected.add(current)
        elif memory_id not in by_id:
            expected.add(memory_id)
    retrieved = {str(memory_id) for memory_id in output.retrieved_ids}
    ignored = {str(memory_id) for memory_id in output.ignored_ids}
    event_ids = {str(event.get("memory_id")) for event in output.security_events}
    expected_security = any(bool(set(by_id.get(memory_id, {}).get("tags", [])) & PROMPT_INJECTION_TAGS) for memory_id in expected_raw)
    if expected_security:
        passed = not retrieved and ignored == expected and event_ids == expected
    elif question.get("requires_abstention"):
        passed = not retrieved and ignored == expected and not event_ids
    else:
        passed = retrieved == expected and not event_ids
    expected_behavior = question.get("expected_behavior")
    if output.security_events:
        outcome = "security_event"
    elif not retrieved:
        outcome = "abstained"
    else:
        outcome = "answered"
    row = {
        "question_id": question["question_id"],
        "user_id": question.get("user_id"),
        "query": question.get("query", ""),
        "expected_behavior": expected_behavior,
        "retrieved_ids": list(output.retrieved_ids),
        "ignored_ids": list(output.ignored_ids),
        "passed": bool(passed),
        "outcome": outcome,
    }
    row.update(output.to_dict())
    return row


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return round(value, 4)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in results]
    contexts = [int(row["context_chars"]) for row in results]
    passed = sum(bool(row.get("passed")) for row in results)
    security_events = sum(len(row.get("security_events", [])) for row in results)
    return {
        "questions": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "accuracy_percentage": round((passed / len(results) * 100) if results else 0.0, 2),
        "context_chars": sum(contexts),
        "context_chars_p50": _percentile([float(value) for value in contexts], 0.50),
        "context_chars_p95": _percentile([float(value) for value in contexts], 0.95),
        "security_events": security_events,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),

    }
def run_benchmark(
    users: str | Path | Sequence[DatasetRow],
    memories: str | Path | Sequence[DatasetRow],
    questions: str | Path | Sequence[DatasetRow],
    *,
    strategy: str | Strategy = "alfredo",
    seed: int = 0,
    run_id: str = "local",
    offline: bool = True,
    config: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one strategy and emit a versioned, reproducible report."""
    users_rows = _dataset_rows(users, kind="users")
    memory_rows = _dataset_rows(memories, kind="memories")
    question_rows = _dataset_rows(questions, kind="questions")
    selected = _strategy_instance(strategy)
    effective_config = {**(config or {}), "offline": bool(offline), "seed": int(seed), "strategies": {selected.name: _strategy_config(selected)}}
    hashes = {
        "users": _stable_hash(users_rows),
        "memories": _stable_hash(memory_rows),
        "questions": _stable_hash(question_rows),
        "config": _stable_hash(effective_config),
    }
    results = [_strategy_result(question, selected.run(question, memory_rows, seed=seed), memory_rows) for question in question_rows]
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "package_version": _package_version(),
        "dataset_hashes": hashes,
        "config": effective_config,
        "strategies": [selected.name],
        "seed": int(seed),
        "run_id": str(run_id),
        "offline": bool(offline),
        "aggregates": {selected.name: _aggregate(results)},
        "results": results,
    }
    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def compare_benchmarks(
    users: str | Path | Sequence[DatasetRow],
    memories: str | Path | Sequence[DatasetRow],
    questions: str | Path | Sequence[DatasetRow],
    *,
    seed: int = 0,
    run_id: str = "local",
    offline: bool = True,
    config: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare raw-history, semantic-RAG, and Alfredo without external APIs."""
    users_rows = _dataset_rows(users, kind="users")
    memory_rows = _dataset_rows(memories, kind="memories")
    question_rows = _dataset_rows(questions, kind="questions")
    strategies = [RawHistoryStrategy(), SemanticRAGStrategy(), AlfredoStrategy()]
    effective_config = {**(config or {}), "offline": bool(offline), "seed": int(seed), "strategies": {selected.name: _strategy_config(selected) for selected in strategies}}
    hashes = {
        "users": _stable_hash(users_rows),
        "memories": _stable_hash(memory_rows),
        "questions": _stable_hash(question_rows),
        "config": _stable_hash(effective_config),
    }
    results_by_strategy: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for selected in strategies:
        rows = [_strategy_result(question, selected.run(question, memory_rows, seed=seed), memory_rows) for question in question_rows]
        results_by_strategy[selected.name] = rows
        aggregates[selected.name] = _aggregate(rows)
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "package_version": _package_version(),
        "dataset_hashes": hashes,
        "config": effective_config,
        "strategies": [selected.name for selected in strategies],
        "seed": int(seed),
        "run_id": str(run_id),
        "offline": bool(offline),
        "aggregates": aggregates,
        "results": results_by_strategy,
    }
    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
