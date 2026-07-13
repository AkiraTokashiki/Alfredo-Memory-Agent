"""Smoke tests for commands documented as the public offline entry points."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memory_agent.cli import commands as commands_module
from memory_agent.cli.commands import cli


class _QuickstartStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _QuickstartAgent:
    """Small deterministic stand-in for the documented offline quickstart."""

    def __init__(self) -> None:
        self.store = _QuickstartStore()
        self.sessions: list[str] = []
        self.perceived: list[str] = []
        self._turn = 0

    def init_session(self, *, label: str) -> None:
        self.sessions.append(label)

    def perceive(self, text: str) -> dict:
        self.perceived.append(text)
        self._turn += 1
        return {
            "recollection_text": (
                "[Retrieved memories]:\n  1. [preference] Python for automation"
                if self._turn == 2
                else ""
            )
        }

    def end_session(self) -> None:
        return None


def test_documented_offline_quickstart_runs_through_cli(monkeypatch) -> None:
    agent = _QuickstartAgent()
    monkeypatch.setattr(commands_module, "_get_agent", lambda _ctx: agent)

    result = CliRunner().invoke(cli, ["--offline", "quickstart"])

    assert result.exit_code == 0, result.output
    assert "Offline quickstart" in result.output
    assert "Remembered:" in result.output
    assert agent.perceived == [
        "I prefer Python for automation",
        "What programming language do I prefer?",
    ]
    assert agent.store.closed is True


def test_documented_benchmark_compare_options_create_offline_report(
    tmp_path: Path,
) -> None:
    users_path = tmp_path / "users.json"
    memories_path = tmp_path / "memories.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    report_path = tmp_path / "comparison.json"

    users_path.write_text(
        json.dumps([{"user_id": "u1", "synthetic": True}]), encoding="utf-8"
    )
    memories_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "memory_id": "m-old",
                    "user_id": "u1",
                    "content": "User previously preferred Python",
                    "confidence": 0.8,
                    "status": "archived",
                    "tags": [],
                },
                {
                    "memory_id": "m-current",
                    "user_id": "u1",
                    "content": "User currently prefers Rust",
                    "confidence": 0.95,
                    "status": "active",
                    "tags": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "user_id": "u1",
                "query": "What language does the user currently prefer?",
                "expected_memory_ids": ["m-current"],
                "expected_behavior": "prefer_updated_memory_over_archived",
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "--offline",
            "benchmark",
            "compare",
            "--users",
            str(users_path),
            "--memories",
            str(memories_path),
            "--questions",
            str(questions_path),
            "--report",
            str(report_path),
            "--seed",
            "42",
            "--run",
            "local-offline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ALFREDO VAULT BENCHMARK COMPARISON" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["seed"] == 42
    assert report["run_id"] == "local-offline"
    assert report["offline"] is True
    assert set(report["strategies"]) == {"raw-history", "semantic-rag", "alfredo"}
