"""Versioned, non-secret Alfredo user configuration."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from memory_agent.core.paths import resolve_memory_home

_SCHEMA_VERSION = 1
_CLIENT_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


class ConfigFormatError(ValueError):
    """Raised when persisted Alfredo configuration is invalid."""


@dataclass(frozen=True)
class RetentionPolicy:
    kind: Literal["forever", "days", "none"] = "days"
    days: int | None = 30

    def __post_init__(self) -> None:
        if self.kind not in {"forever", "days", "none"}:
            raise ValueError("retention kind must be forever, days, or none")
        if self.kind == "days" and (not isinstance(self.days, int) or isinstance(self.days, bool) or self.days <= 0):
            raise ValueError("retention days must be a positive integer")
        if self.kind != "days" and self.days is not None:
            raise ValueError("retention days must be None unless kind is days")


@dataclass
class AlfredoUserConfig:
    schema_version: int = _SCHEMA_VERSION
    onboarding_complete: bool = False
    default_namespace: str | None = None
    default_user_id: str | None = None
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    mcp_clients: dict[str, dict[str, str | bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION or not isinstance(self.schema_version, int):
            raise ValueError("unsupported schema_version")
        if not isinstance(self.onboarding_complete, bool):
            raise ValueError("onboarding_complete must be boolean")
        for field_name in ("default_namespace", "default_user_id"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be string or None")
        if not isinstance(self.mcp_clients, dict):
            raise ValueError("mcp_clients must be an object")
        for name, record in self.mcp_clients.items():
            if not isinstance(name, str) or _CLIENT_NAME.fullmatch(name) is None:
                raise ValueError("MCP client names must match [A-Za-z0-9._-]+")
            if not isinstance(record, dict) or set(record) - {"approved", "fingerprint"}:
                raise ValueError("invalid MCP client record")
            if "approved" in record and not isinstance(record["approved"], bool):
                raise ValueError("MCP approved must be boolean")
            if "fingerprint" in record and (
                not isinstance(record["fingerprint"], str) or _FINGERPRINT.fullmatch(record["fingerprint"]) is None
            ):
                raise ValueError("MCP fingerprint must match sha256:<64 lowercase hex>")


def user_config_path(home: Path | None = None) -> Path:
    root = Path(home) if home is not None else resolve_memory_home()
    return root / "config.json"


def load_user_config(home: Path | None = None) -> AlfredoUserConfig:
    path = user_config_path(home)
    if not path.exists():
        return AlfredoUserConfig()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        expected = {"schema_version", "onboarding_complete", "default_namespace", "default_user_id", "retention", "mcp_clients"}
        if not isinstance(document, dict) or set(document) != expected:
            raise ConfigFormatError("configuration keys do not match supported schema")
        retention = document["retention"]
        if not isinstance(retention, dict) or set(retention) != {"kind", "days"}:
            raise ConfigFormatError("retention must contain kind and days")
        return AlfredoUserConfig(
            schema_version=document["schema_version"],
            onboarding_complete=document["onboarding_complete"],
            default_namespace=document["default_namespace"],
            default_user_id=document["default_user_id"],
            retention=RetentionPolicy(**retention),
            mcp_clients=document["mcp_clients"],
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigFormatError):
            raise
        raise ConfigFormatError(f"unable to read user configuration: {exc}") from exc


def save_user_config(config: AlfredoUserConfig, home: Path | None = None) -> Path:
    if not isinstance(config, AlfredoUserConfig):
        raise TypeError("config must be AlfredoUserConfig")
    document = {
        "schema_version": config.schema_version,
        "onboarding_complete": config.onboarding_complete,
        "default_namespace": config.default_namespace,
        "default_user_id": config.default_user_id,
        "retention": {"kind": config.retention.kind, "days": config.retention.days},
        "mcp_clients": config.mcp_clients,
    }
    path = user_config_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".config.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path
