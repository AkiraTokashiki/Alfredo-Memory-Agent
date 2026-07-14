"""Black-box contracts for CI, release, and GitHub contribution surfaces.

These checks inspect repository-owned workflow and template files without invoking
GitHub, PyPI, or any other network service.  YAML is parsed when PyYAML is
available so trigger and matrix assertions are semantic; the text fallback
keeps the contract runnable in a minimal test environment.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

try:  # PyYAML is optional for the repository's minimal test environment.
    import yaml
except ImportError:  # pragma: no cover - exercised only without the optional parser.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"


def _required_file(relative_path: str) -> tuple[Path, str]:
    """Read a repository file while turning absence into a contract failure."""

    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.fail(f"required GitHub surface is missing: {relative_path}")
    return path, path.read_text(encoding="utf-8")


def _workflow(relative_path: str) -> tuple[Path, str, dict[str, Any] | None]:
    path, text = _required_file(relative_path)
    if yaml is None:
        return path, text, None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        pytest.fail(f"{relative_path} must be valid YAML: {exc}")
    if not isinstance(parsed, dict):
        pytest.fail(f"{relative_path} must contain a YAML mapping")
    return path, text, parsed


def _yaml_on(workflow: dict[str, Any]) -> Any:
    """Return the workflow's ``on`` mapping, accounting for YAML 1.1 parsing."""

    for key, value in workflow.items():
        # PyYAML 5/6 resolves the unquoted YAML 1.1 key ``on`` to True.
        if key == "on" or key is True:
            return value
    return None


def _event_has_main(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    branches = event.get("branches", [])
    if isinstance(branches, str):
        branches = [branches]
    return isinstance(branches, list) and "main" in {str(branch) for branch in branches}


def _mapping_values(node: Any, wanted_key: str) -> list[Any]:
    """Collect values for a key from nested parsed YAML mappings."""

    found: list[Any] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) == wanted_key:
                found.append(value)
            found.extend(_mapping_values(value, wanted_key))
    elif isinstance(node, list):
        for value in node:
            found.extend(_mapping_values(value, wanted_key))
    return found


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _flatten_values(nested)]
    return [str(value)]


def _assert_build_and_twine(text: str, surface: str) -> None:
    assert re.search(r"(?:python\s+-m\s+)?build\b", text, re.IGNORECASE), (
        f"{surface} must build a distribution"
    )
    assert re.search(r"(?:python\s+-m\s+)?twine\s+check\b", text, re.IGNORECASE), (
        f"{surface} must validate built metadata with twine check"
    )


def _template_text(kind: str) -> tuple[Path, str]:
    if not ISSUE_TEMPLATE_DIR.is_dir():
        pytest.fail("required GitHub issue-template directory is missing")
    candidates = sorted(
        path
        for path in ISSUE_TEMPLATE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".yaml", ".yml"}
    )
    matches = [path for path in candidates if kind in path.stem.casefold()]
    if not matches:
        pytest.fail(f"missing {kind} issue template in {ISSUE_TEMPLATE_DIR}")
    path = matches[0]
    return path, path.read_text(encoding="utf-8")


def test_ci_workflow_is_main_matrix_minimal_and_offline() -> None:
    """CI runs the supported Python matrix without credentials or model downloads."""

    _, text, parsed = _workflow(".github/workflows/ci.yml")
    if parsed is not None:
        events = _yaml_on(parsed)
        assert isinstance(events, dict), "ci.yml must define mapping-style workflow triggers"
        assert _event_has_main(events.get("push")), "CI push trigger must target main"
        assert _event_has_main(events.get("pull_request")), "CI PR trigger must target main"

        versions = {
            version
            for matrix in _mapping_values(parsed, "python-version")
            for version in _flatten_values(matrix)
        }
        assert {"3.11", "3.12", "3.13", "3.14"} <= versions
    else:
        for event in ("push", "pull_request"):
            event_match = re.search(rf"(?ms)^\s*{event}:\s*$.*?(?=^\S|\Z)", text)
            assert event_match and re.search(r"branches:[^\n]*\n\s*-\s*main\b", event_match.group()), (
                f"CI {event} trigger must target main"
            )
        for version in ("3.11", "3.12", "3.13", "3.14"):
            assert re.search(rf"['\"]?{re.escape(version)}['\"]?", text)

    install_lines = [line for line in text.splitlines() if re.search(r"pip\s+install", line, re.IGNORECASE)]
    assert any(
        re.search(r"(?:-e|--editable)\s+\.\s*(?:[#;&|]|$)", line, re.IGNORECASE)
        and not re.search(r"\.\[[^\]]+\]", line)
        for line in install_lines
    ), "CI must install the project editable with its minimal dependency set"
    assert re.search(r"offline", text, re.IGNORECASE), "CI must exercise offline behavior"
    assert re.search(r"test_documentation|documentation.*test|test.*documentation", text, re.IGNORECASE), (
        "CI must run documentation tests"
    )
    assert re.search(r"(?:python\s+-m\s+)?pytest\b", text, re.IGNORECASE), "CI must run pytest"
    assert any(
        re.search(r"(?:python\s+-m\s+)?pytest\b", line, re.IGNORECASE)
        and not re.search(r"tests/test_[\w.-]+\.py", line, re.IGNORECASE)
        for line in text.splitlines()
    ), "CI must include one complete-suite pytest command"
    _assert_build_and_twine(text, "ci.yml")

    assert not re.search(r"api[_-]?key", text, re.IGNORECASE), "CI must not require API keys"
    assert not re.search(
        r"huggingface|sentence[-_ ]transformers?|from_pretrained|"
        r"(?:download|fetch).{0,40}\bmodel\b|\bmodel\b.{0,40}(?:download|fetch)",
        text,
        re.IGNORECASE | re.DOTALL,
    ), "CI must not download or initialize external models"


def test_release_workflow_uses_tagged_oidc_trusted_publishing() -> None:
    """Releases build v-tags and publish through PyPI's tokenless OIDC flow."""

    _, text, parsed = _workflow(".github/workflows/release.yml")
    if parsed is not None:
        events = _yaml_on(parsed)
        assert isinstance(events, dict), "release.yml must define mapping-style workflow triggers"
        push = events.get("push")
        assert isinstance(push, dict), "release workflow must trigger from push tags"
        tags = push.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        assert any(str(tag).startswith("v") and "*" in str(tag) for tag in tags)
    else:
        assert re.search(r"(?ms)^\s*push:\s*$.*?^\s*tags:\s*$.*?[-]\s*[\"']?v\*", text), (
            "release workflow must trigger on v* tags"
        )

    _assert_build_and_twine(text, "release.yml")
    assert "alfredo-memory-agent" in text
    assert re.search(r"id-token\s*:\s*write", text, re.IGNORECASE), (
        "release publishing must request GitHub's OIDC id-token permission"
    )
    assert re.search(r"pypa/gh-action-pypi-publish", text, re.IGNORECASE), (
        "release must use PyPI's trusted-publishing action"
    )
    assert not re.search(r"PYPI_API_TOKEN", text, re.IGNORECASE)
    assert not re.search(r"\bpassword\s*:", text, re.IGNORECASE), (
        "release must not configure a password/token publishing credential"
    )
    assert not re.search(r"secrets\.[A-Z0-9_]*(?:TOKEN|PASSWORD|API_KEY)", text, re.IGNORECASE)


def test_github_issue_templates_capture_actionable_reports() -> None:
    """Bug and feature forms request enough information to act on a report."""

    _, bug = _template_text("bug")
    bug_normalized = bug.casefold()
    assert re.search(r"reproduc|steps?\s+to\s+reproduce", bug_normalized)
    assert re.search(r"expected", bug_normalized)
    assert re.search(r"actual|observed", bug_normalized)
    assert re.search(r"environment|python|operating\s+system|version", bug_normalized)

    _, feature = _template_text("feature")
    feature_normalized = feature.casefold()
    assert re.search(r"problem|need|motivat", feature_normalized)
    assert re.search(r"proposed|solution|describe.*feature|request", feature_normalized)
    assert re.search(r"alternative|workaround|considered", feature_normalized)


def test_pull_request_template_requires_summary_tests_and_checklist() -> None:
    """Pull requests disclose their change, verification, and contributor checklist."""

    github_dir = REPO_ROOT / ".github"
    candidates: list[Path] = []
    canonical = github_dir / "PULL_REQUEST_TEMPLATE.md"
    if canonical.is_file():
        candidates.append(canonical)
    if github_dir.is_dir():
        candidates.extend(
            path
            for path in github_dir.iterdir()
            if path.is_file() and "pull_request_template" in path.name.casefold()
        )
    if not candidates:
        pytest.fail("missing .github pull request template")
    text = candidates[0].read_text(encoding="utf-8").casefold()
    assert re.search(r"summary|description|what changed|change", text)
    assert re.search(r"test|verification|how tested", text)
    assert re.search(r"^\s*[-*]\s*\[\s*[x ]?\s*\]", text, re.MULTILINE), (
        "PR template must include Markdown checklist boxes"
    )

@pytest.mark.parametrize("directory_name", ("dist", "build"))
def test_repository_does_not_ship_distribution_artifacts(directory_name: str) -> None:
    """If generated distribution directories exist, none may be versioned."""

    directory = REPO_ROOT / directory_name
    if not directory.exists():
        return
    assert directory.is_dir(), f"{directory_name} must not be a committed file"
    result = subprocess.run(
        ["git", "ls-files", "--", f"{directory_name}/"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert not tracked, f"generated {directory_name} artifacts must not be tracked: {tracked}"
