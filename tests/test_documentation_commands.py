"""Smoke tests for commands documented as the public offline entry points."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
        self.ended_sessions: list[str] = []
        self.perceived: list[str] = []
        self._active_label = ""
        self._turn = 0

    def init_session(self, *, label: str) -> None:
        self.sessions.append(label)
        self._active_label = label

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
        self.ended_sessions.append(self._active_label)
        self._active_label = ""


def test_documented_offline_quickstart_runs_through_cli(monkeypatch) -> None:
    agent = _QuickstartAgent()
    monkeypatch.setattr(commands_module, "_get_agent", lambda _ctx: agent)

    result = CliRunner().invoke(cli, ["--offline", "quickstart"])

    assert result.exit_code == 0, result.output
    assert "Alfredo MemoryAgent" in result.output
    assert "Alfredo MemoryAgent — offline quickstart" in result.output
    assert "Persistent memory • timely forgetting • bounded, explainable recall" in result.output
    assert "Remembered:" in result.output
    assert agent.perceived == [
        "I prefer Python for automation",
        "What programming language do I prefer?",
    ]
    assert agent.sessions == ["quickstart", "quickstart-recall"]
    assert agent.ended_sessions == ["quickstart", "quickstart-recall"]
    assert agent.store.closed is True


def test_lifecycle_demo_has_stable_four_stage_output() -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "examples" / "demo_lifecycle.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    output = result.stdout
    markers = [
        "[1] learn preference",
        "[2] recall across session",
        "[3] supersede stale preference",
        "[4] bounded context and trust evidence",
    ]
    positions = [output.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert all(output.count(marker) == 1 for marker in markers)

    learn_end = output.index(markers[1])
    recall_end = output.index(markers[2])
    supersede_end = output.index(markers[3])
    assert "The user prefers: python for automation" in output[:learn_end]
    assert "recall: The user prefers: python for automation" in output[learn_end:recall_end]
    assert "consolidation=update" in output[recall_end:supersede_end]
    assert "The user does not like: python" in output[recall_end:supersede_end]
    assert "archived=1" in output[recall_end:supersede_end]
    bounded = output[supersede_end:]
    selected_line = next(
        line for line in bounded.splitlines() if line.startswith("packet selected: ids=")
    )
    omitted_line = next(
        line for line in bounded.splitlines() if line.startswith("packet omitted: ids=")
    )
    selected_ids = ast.literal_eval(selected_line.split("ids=", 1)[1].split(";", 1)[0])
    omitted_ids = ast.literal_eval(omitted_line.split("ids=", 1)[1].split(";", 1)[0])
    assert selected_ids
    assert omitted_ids
    trust_line = next(line for line in bounded.splitlines() if line.startswith("trust evidence:"))
    assert "trusted" in trust_line
    assert "untrusted" in trust_line
    assert "new memories:" in output
    assert "recall:" in output
    assert "lifecycle:" in output
    assert "created_at" not in output


def test_lifecycle_demo_closes_active_session_when_perceive_fails(monkeypatch) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "examples" / "demo_lifecycle.py"
    spec = importlib.util.spec_from_file_location("demo_lifecycle_failure", script)
    assert spec is not None and spec.loader is not None
    demo_lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo_lifecycle)
    events: list[str] = []

    class FailingAgent:
        def __init__(self, **_kwargs) -> None:
            events.append("construct")

        def init_session(self, _label: str) -> None:
            events.append("init")

        def perceive(self, _text: str) -> dict:
            events.append("perceive")
            raise RuntimeError("primary failure")

        def end_session(self) -> None:
            events.append("end")

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close cleanup failure")
    monkeypatch.setattr(demo_lifecycle, "MemoryAgent", FailingAgent)
    with pytest.raises(RuntimeError, match="primary failure"):
        demo_lifecycle.main()
    assert events == ["construct", "init", "perceive", "end", "close"]


def test_basic_demo_runs_without_remote_model_download() -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "examples" / "demo_basic.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "Alfredo MemoryAgent — Basic Demo" in result.stdout
    assert "Final statistics:" in result.stdout
    assert "Demo complete." in result.stdout
    assert "HF Hub" not in result.stdout + result.stderr
    assert "Loading weights" not in result.stdout + result.stderr


def test_hackathon_demo_uses_isolated_sqlite_without_remote_model(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "examples" / "demo_hackathon.py"
    env = os.environ.copy()
    env["ALFREDO_HOME"] = str(tmp_path / "alfredo-home")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "Alfredo MemoryAgent — Hackathon Demo" in result.stdout
    assert "Session 1: learn preferences" in result.stdout
    assert "Session 4: bounded recall after noise" in result.stdout
    assert "=== Stats ===" in result.stdout
    assert "HF Hub" not in result.stdout + result.stderr
    assert "Loading weights" not in result.stdout + result.stderr
    assert not list(tmp_path.rglob("hackathon_demo.db"))


@pytest.mark.parametrize("demo_filename", ["demo_basic.py", "demo_hackathon.py"])
def test_existing_demos_preserve_primary_error_during_cleanup(
    monkeypatch, demo_filename: str
) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "examples" / demo_filename
    module_name = f"failure_{demo_filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    events: list[str] = []

    class FailingAgent:
        def __init__(self, **_kwargs) -> None:
            events.append("construct")

        def init_session(self, *_args, **_kwargs) -> None:
            events.append("init")

        def perceive(self, _text: str) -> dict:
            events.append("perceive")
            raise RuntimeError("primary failure")

        def end_session(self) -> None:
            events.append("end")
            raise RuntimeError("end cleanup failure")

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("store cleanup failure")

    monkeypatch.setattr(demo, "MemoryAgent", FailingAgent)
    with pytest.raises(RuntimeError, match="primary failure"):
        demo.main()
    assert events == ["construct", "init", "perceive", "end", "close"]


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


def test_requirements_bootstrap_local_semantic_distribution() -> None:
    requirements = Path(__file__).parents[1] / "requirements.txt"

    assert "-e .[semantic]" in requirements.read_text(encoding="utf-8").splitlines()
