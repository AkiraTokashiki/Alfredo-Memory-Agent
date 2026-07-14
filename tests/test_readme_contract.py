"""Black-box adoption contracts for the public README and lifecycle visual."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
LIFECYCLE_ASSET = REPO_ROOT / "docs" / "assets" / "alfredo-memory-lifecycle.svg"


@pytest.fixture(scope="module")
def readme() -> str:
    """Load the public README once for this focused documentation contract."""

    return README_PATH.read_text(encoding="utf-8")


def _fenced_command_blocks(markdown: str) -> list[str]:
    """Return executable fenced blocks while ignoring prose and inline examples."""

    return re.findall(
        r"```(?:bash|console|powershell|shell|sh)?\s*\n(.*?)```",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _command_lines(markdown: str) -> list[str]:
    return [
        line.strip()
        for block in _fenced_command_blocks(markdown)
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _local_link_targets(markdown: str) -> set[str]:
    targets: set[str] = set()
    for raw_target in re.findall(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)", markdown):
        target = raw_target.strip().split("#", 1)[0]
        if not target or re.match(r"[a-z][a-z0-9+.-]*://", target, re.IGNORECASE):
            continue
        targets.add(target.removeprefix("./").rstrip("/"))
    return targets


def _benchmark_sections(markdown: str) -> list[str]:
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", markdown, re.MULTILINE))
    sections: list[str] = []
    for index, heading in enumerate(headings):
        if "benchmark" not in heading.group(2).casefold():
            continue
        level = len(heading.group(1))
        end = len(markdown)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        sections.append(markdown[heading.start() : end])
    if not sections:
        pytest.fail("README must have a heading for benchmark evidence")
    return sections


def test_readme_hero_positions_alfredo_as_a_memory_agent(readme: str) -> None:
    """The first impression names the product and its four lifecycle promises."""

    hero = re.split(r"^---\s*$", readme, maxsplit=1, flags=re.MULTILINE)[0]
    assert re.search(r"\bAlfredo\b", hero)
    assert re.search(r"\bMemoryAgent\b", hero)
    for capability in ("learn", "remember", "forget", "explain"):
        assert re.search(rf"\b{capability}\w*\b", hero, re.IGNORECASE), capability


def test_readme_publishes_the_canonical_install_and_offline_quickstart(readme: str) -> None:
    """A fresh user can copy the two supported commands without editable extras."""

    commands = _command_lines(readme)
    assert "pip install alfredo-memory-agent" in commands
    assert "python -m memory_agent --offline quickstart" in commands


def test_readme_links_the_local_adoption_paths(readme: str) -> None:
    """The funnel exposes demo, integration, evidence, and community entry points."""

    targets = _local_link_targets(readme)
    expected = {
        "examples/demo_lifecycle.py",
        "INTEGRATION.md",
        "benchmarks/alfredos_vault",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
    }
    assert expected <= targets


def test_readme_limits_benchmark_claims_to_synthetic_evidence(readme: str) -> None:
    """Benchmark visibility is paired with an explicit production-security limit."""

    limitation = re.compile(
        r"\b(?:not|does\s+not|doesn't|cannot|can't|isn't)\b"
        r".{0,120}\b(?:security|privacy)\b",
        flags=re.IGNORECASE | re.DOTALL,
    )
    sections = _benchmark_sections(readme)
    assert any(re.search(r"\bsynthetic\b", section, re.IGNORECASE) for section in sections)
    assert any(limitation.search(section) for section in sections), (
        "benchmark section must not imply a production security/privacy audit"
    )


def test_lifecycle_svg_exists_with_readable_lifecycle_labels() -> None:
    """The hero visual remains inspectable without depending on a browser renderer."""

    assert LIFECYCLE_ASSET.is_file(), f"missing lifecycle visual: {LIFECYCLE_ASSET}"
    root = ET.fromstring(LIFECYCLE_ASSET.read_text(encoding="utf-8"))
    assert root.tag.rsplit("}", 1)[-1] == "svg"
    visible_labels = " ".join(
        (element.text or "")
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    ).casefold()
    assert visible_labels, "lifecycle SVG must contain readable text labels"
    for label in ("learn", "retrieve", "trust", "pack", "reinforce"):
        assert re.search(rf"\b{label}\w*\b", visible_labels), label
    assert re.search(r"\b(?:supersed\w*|forget\w*)\b", visible_labels)


def test_rejected_distribution_names_are_not_executable_install_commands(readme: str) -> None:
    """Historical/provenance prose may mention collisions, but copyable commands may not."""

    prohibited = re.compile(
        r"^(?:[$>]\s*)?(?:python\s+-m\s+)?pip\s+install\s+"
        r"(?:['\"]?)(?:alfredo|memory-agent)(?:['\"]?)(?:\s|$)",
        re.IGNORECASE,
    )
    bad_commands = [line for line in _command_lines(readme) if prohibited.match(line)]
    assert not bad_commands, bad_commands
