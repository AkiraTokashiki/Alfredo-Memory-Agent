"""Native filesystem paths for Alfredo runtime state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Alfredo"
DB_FILENAME = "memory_agent.db"


def _is_dev_repo(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return 'name = "alfredo-memory-agent"' in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


def _package_project_root() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    return root if _is_dev_repo(root) else None


def _os_app_data_home() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    return Path.home() / ".alfredo"


def resolve_memory_home(project_root: Path | None = None) -> Path:
    """Return Alfredo's native memory directory and ensure it exists."""
    env_home = os.environ.get("ALFREDO_HOME")
    if env_home:
        home = Path(env_home).expanduser()
    else:
        root = project_root or _package_project_root()
        home = root / ".alfredo" if root and _is_dev_repo(root) else _os_app_data_home()

    home.mkdir(parents=True, exist_ok=True)
    return home


def default_memory_db_path() -> Path:
    """Return the default SQLite DB path for Alfredo memory."""
    return resolve_memory_home() / DB_FILENAME
