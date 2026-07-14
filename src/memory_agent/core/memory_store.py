"""SQLite-backed persistent memory storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_agent.models import (
    EvolutionDecision,
    EvolutionProposal,
    MemoryRecord,
    MemoryRelation,
    SessionRecord,
)
from memory_agent.core.relations import RelationManager


class MemoryStore:
    """Persistent memory storage backed by SQLite.

    Manages memories, embeddings, tags, sessions, and the
    many-to-many relationships between them.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        memory_type TEXT NOT NULL DEFAULT 'episodic',
        importance REAL NOT NULL DEFAULT 0.5,
        strength REAL NOT NULL DEFAULT 1.0,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        metadata TEXT DEFAULT '{}',
        is_active INTEGER NOT NULL DEFAULT 1,
        namespace TEXT,
        confidence REAL,
        sensitivity TEXT,
        source TEXT,
        superseded_by INTEGER,
        last_decision_reason TEXT
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        memory_id INTEGER PRIMARY KEY,
        vector BLOB NOT NULL,
        model_name TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS memory_tags (
        memory_id INTEGER NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (memory_id, tag),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT,
        namespace TEXT
    );

    CREATE TABLE IF NOT EXISTS memory_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        relation_type TEXT NOT NULL DEFAULT 'related_to',
        confidence REAL NOT NULL DEFAULT 1.0,
        namespace TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        source TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
    );
    
    CREATE TABLE IF NOT EXISTS memory_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        actor TEXT NOT NULL,
        namespace TEXT,
        memory_id INTEGER,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS session_memories (
        session_id INTEGER NOT NULL,
        memory_id INTEGER NOT NULL,
        turn_index INTEGER,
        PRIMARY KEY (session_id, memory_id),
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at);
    CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active) WHERE is_active = 1;
    CREATE INDEX IF NOT EXISTS idx_session_memories_session ON session_memories(session_id);
    """

    SCHEMA_VERSION = 4

    def initialize(self) -> None:
        """Create tables and apply idempotent, rollback-safe migrations."""
        self.conn.executescript(self.SCHEMA_SQL)
        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            columns_by_table = {
                "memories": {
                    "namespace": "TEXT",
                    "confidence": "REAL",
                    "sensitivity": "TEXT",
                    "source": "TEXT",
                    "superseded_by": "INTEGER",
                    "last_decision_reason": "TEXT",
                },
                "sessions": {"namespace": "TEXT"},
                "memory_relations": {
                    "source_id": "INTEGER",
                    "target_id": "INTEGER",
                    "relation_type": "TEXT DEFAULT 'related_to'",
                    "confidence": "REAL DEFAULT 1.0",
                    "namespace": "TEXT",
                    "created_at": "TEXT",
                    "updated_at": "TEXT",
                    "source": "TEXT",
                    "is_active": "INTEGER NOT NULL DEFAULT 1",
                },
            }
            for table, columns in columns_by_table.items():
                existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                for column, definition in columns.items():
                    if column not in existing:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

            self._migrate_relation_table()
            self.conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(self.SCHEMA_VERSION),),
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_source ON memory_relations(source_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relations_target ON memory_relations(target_id)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_relations_active "
                "ON memory_relations(is_active) WHERE is_active = 1"
            )
            self.conn.execute("DROP INDEX IF EXISTS uq_memory_relations_edge")
            self.conn.execute(
                "CREATE UNIQUE INDEX uq_memory_relations_edge "
                "ON memory_relations(source_id, target_id, relation_type, COALESCE(namespace, ''))"
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _migrate_relation_table(self) -> None:
        """Upgrade legacy relation tables, dropping invalid rows and duplicates."""
        foreign_keys = {
            (row[3], row[2], row[4], row[6].upper())
            for row in self.conn.execute("PRAGMA foreign_key_list(memory_relations)")
        }
        required_foreign_keys = {
            ("source_id", "memories", "id", "CASCADE"),
            ("target_id", "memories", "id", "CASCADE"),
        }
        if required_foreign_keys <= foreign_keys:
            return
        self.conn.execute("DROP INDEX IF EXISTS uq_memory_relations_edge")
        self.conn.execute("DROP INDEX IF EXISTS idx_memory_relations_source")
        self.conn.execute("DROP INDEX IF EXISTS idx_memory_relations_target")
        self.conn.execute("DROP INDEX IF EXISTS idx_memory_relations_active")
        self.conn.execute("ALTER TABLE memory_relations RENAME TO memory_relations_legacy")
        self.conn.execute(
            """CREATE TABLE memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related_to',
                confidence REAL NOT NULL DEFAULT 1.0,
                namespace TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                source TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            )"""
        )
        legacy_rows = self.conn.execute(
            "SELECT source_id, target_id, relation_type, confidence, namespace, "
            "created_at, updated_at, source, is_active FROM memory_relations_legacy"
        ).fetchall()
        deduped: dict[tuple[Any, Any, str, Any], tuple[Any, ...]] = {}
        for row in legacy_rows:
            source_id, target_id = row["source_id"], row["target_id"]
            if source_id is None or target_id is None:
                continue
            source_memory = self.conn.execute(
                "SELECT namespace FROM memories WHERE id = ?", (source_id,)
            ).fetchone()
            target_memory = self.conn.execute(
                "SELECT namespace FROM memories WHERE id = ?", (target_id,)
            ).fetchone()
            if source_memory is None or target_memory is None:
                continue
            if source_memory["namespace"] != target_memory["namespace"]:
                continue
            row_namespace = row["namespace"]
            if row_namespace is not None and row_namespace != source_memory["namespace"]:
                continue
            relation_type = row["relation_type"] or "related_to"
            row_namespace = row_namespace if row_namespace is not None else source_memory["namespace"]
            key = (source_id, target_id, relation_type, row_namespace)
            if key not in deduped:
                deduped[key] = (
                    source_id, target_id, relation_type,
                    row["confidence"] if row["confidence"] is not None else 1.0,
                    row_namespace, row["created_at"], row["updated_at"], row["source"],
                    row["is_active"] if row["is_active"] is not None else 1,
                )
        self.conn.executemany(
            """INSERT INTO memory_relations
                (source_id, target_id, relation_type, confidence, namespace,
                 created_at, updated_at, source, is_active)
            VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')),
                    COALESCE(?, datetime('now')), ?, ?)""",
            list(deduped.values()),
        )
        self.conn.execute("DROP TABLE memory_relations_legacy")
    def commit(self) -> None:
        """Commit pending lifecycle mutations."""
        self.conn.commit()

    def apply_evolution(self, proposal: EvolutionProposal) -> EvolutionDecision:
        """Validate and apply one proposal, recording exactly one audit event."""
        if not isinstance(proposal, EvolutionProposal):
            raise TypeError("proposal must be an EvolutionProposal")

        savepoint = "memory_evolution"
        self.conn.execute(f"SAVEPOINT {savepoint}")
        try:
            rejection = self._validate_evolution(proposal)
            if rejection is not None:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                event_id = self._record_evolution_event(
                    "evolution_rejected", proposal, rejection
                )
                self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                self.conn.commit()
                return EvolutionDecision(False, proposal, rejection, event_id)

            namespace = proposal.namespace
            candidate = self.get_memory(proposal.candidate_id, namespace=namespace)
            if candidate is None:
                raise ValueError("candidate memory not found")
            relation_confidence = proposal.confidence
            for target_id in proposal.target_ids:
                relation = MemoryRelation(
                    source_id=proposal.candidate_id,
                    target_id=target_id,
                    relation_type=proposal.relation_type,
                    confidence=relation_confidence,
                    namespace=namespace,
                    source=proposal.actor,
                )
                self.add_relation(relation, namespace=namespace, commit=False)

            candidate.metadata = {
                **candidate.metadata,
                **dict(proposal.metadata_patch),
            }
            self.update_memory(candidate, namespace=namespace, commit=False)
            for target_id in proposal.target_ids:
                self.archive_memory(
                    target_id,
                    reason="superseded by accepted evolution",
                    metadata={"superseded_by": proposal.candidate_id},
                    namespace=namespace,
                    commit=False,
                )
                target = self.get_memory(target_id, namespace=namespace)
                if target is None:
                    raise ValueError("target memory disappeared during evolution")
                target.superseded_by = proposal.candidate_id
                target.last_decision_reason = proposal.reason
                self.update_memory(target, namespace=namespace, commit=False)

            event_id = self._record_evolution_event(
                "evolution_accepted", proposal, proposal.reason
            )
            self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            self.conn.commit()
            return EvolutionDecision(True, proposal, "proposal applied", event_id)
        except Exception as exc:
            self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            rejection = self._safe_rejection_reason(exc)
            event_id = self._record_evolution_event(
                "evolution_rejected", proposal, rejection
            )
            self.conn.commit()
            return EvolutionDecision(False, proposal, rejection, event_id)

    def _validate_evolution(self, proposal: EvolutionProposal) -> str | None:
        """Return a stable rejection reason without exposing proposal content."""
        import math

        if proposal.action not in {"supersede"}:
            return "rejected: unsupported evolution action"
        try:
            RelationManager.validate_type(proposal.relation_type)
        except (TypeError, ValueError):
            return "rejected: unsupported relation type"
        if proposal.action == "supersede" and proposal.relation_type != "supersedes":
            return "rejected: supersede action requires supersedes relation"
        try:
            if not math.isfinite(proposal.confidence):
                return "rejected: confidence must be finite"
        except (TypeError, ValueError):
            return "rejected: confidence must be finite"
        try:
            proposal.to_dict()
        except (TypeError, ValueError):
            return "rejected: proposal payload is not JSON-safe"
        if proposal.evidence.trust != "trusted":
            return "rejected: evidence is not trusted"
        if not proposal.target_ids:
            return "rejected: proposal has no targets"
        if len(set(proposal.target_ids)) != len(proposal.target_ids):
            return "rejected: proposal contains duplicate targets"
        if any(isinstance(item, bool) or not isinstance(item, int) for item in proposal.target_ids):
            return "rejected: target ids are invalid"
        if isinstance(proposal.candidate_id, bool) or not isinstance(proposal.candidate_id, int):
            return "rejected: candidate id is invalid"
        candidate = self.get_memory(proposal.candidate_id, namespace=proposal.namespace)
        if candidate is None:
            return "rejected: candidate namespace or id mismatch"
        if not candidate.is_active:
            return "rejected: candidate memory is archived"
        for target_id in proposal.target_ids:
            if target_id == proposal.candidate_id:
                return "rejected: candidate cannot target itself"
            target = self.get_memory(target_id, namespace=proposal.namespace)
            if target is None:
                return "rejected: target namespace or id mismatch"
            if not target.is_active:
                return "rejected: target memory is archived"
        try:
            json.dumps(proposal.metadata_patch, allow_nan=False)
        except (TypeError, ValueError):
            return "rejected: metadata patch is not JSON-safe"
        return None

    @staticmethod
    def _safe_rejection_reason(exc: Exception) -> str:
        del exc
        return "rejected: evolution transaction failed"

    def _record_evolution_event(
        self, event_type: str, proposal: EvolutionProposal, reason: str
    ) -> int:
        import re

        actor_value = str(proposal.actor)
        actor = (
            actor_value
            if re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", actor_value)
            else "unknown"
        )
        exists = self.conn.execute(
            "SELECT 1 FROM memories WHERE id = ?", (proposal.candidate_id,)
        ).fetchone()
        memory_id = proposal.candidate_id if exists is not None else None
        cur = self.conn.execute(
            "INSERT INTO memory_events "
            "(event_type, actor, namespace, memory_id, reason) VALUES (?, ?, ?, ?, ?)",
            (event_type, actor, proposal.namespace, memory_id, reason),
        )
        return int(cur.lastrowid)
    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def add_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        """Insert a new memory record. Returns the new row id."""
        if namespace is not None:
            memory.namespace = namespace
        now = datetime.now().isoformat()
        cur = self.conn.execute(
            """INSERT INTO memories
               (content, memory_type, importance, strength,
                access_count, last_accessed_at, created_at, updated_at,
                metadata, is_active, namespace, confidence, sensitivity,
                source, superseded_by, last_decision_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.strength,
                memory.access_count,
                memory.last_accessed_at or now,
                memory.created_at or now,
                now,
                json.dumps(memory.metadata),
                1 if memory.is_active else 0,
                memory.namespace,
                memory.confidence,
                memory.sensitivity,
                memory.source,
                memory.superseded_by,
                memory.last_decision_reason,
            ),
        )
        memory_id = cur.lastrowid
        memory.id = memory_id

        for tag in memory.tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )

        if commit:
            self.conn.commit()
        return memory_id

    def get_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> MemoryRecord | None:
        """Fetch a single memory, optionally constrained to a namespace."""
        query = "SELECT * FROM memories WHERE id = ?"
        params: list[Any] = [memory_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Update an existing memory, optionally constrained to a namespace."""
        now = datetime.now().isoformat()
        stored_namespace = namespace
        if namespace is None and memory.id is not None:
            existing = self.conn.execute(
                "SELECT namespace FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
            if existing is not None:
                stored_namespace = existing["namespace"]
        query = """UPDATE memories SET
               content = ?, memory_type = ?, importance = ?, strength = ?,
               access_count = ?, last_accessed_at = ?, updated_at = ?,
               metadata = ?, is_active = ?, namespace = ?, confidence = ?,
               sensitivity = ?, source = ?, superseded_by = ?,
               last_decision_reason = ?
               WHERE id = ?"""
        params: list[Any] = [
            memory.content,
            memory.memory_type,
            memory.importance,
            memory.strength,
            memory.access_count,
            memory.last_accessed_at or now,
            now,
            json.dumps(memory.metadata),
            1 if memory.is_active else 0,
            stored_namespace,
            memory.confidence,
            memory.sensitivity,
            memory.source,
            memory.superseded_by,
            memory.last_decision_reason,
            memory.id,
        ]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        cur = self.conn.execute(query, params)
        if cur.rowcount and memory.id is not None:
            self.conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory.id,))
            for tag in memory.tags:
                self.conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (memory.id, tag),
                )
        if commit:
            self.conn.commit()

    def delete_memory(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        namespace: str | None = None,
    ) -> None:
        """Soft-delete or hard-delete a memory in an optional namespace."""
        predicate = "id = ?"
        params: list[Any] = [memory_id]
        if namespace is not None:
            predicate += " AND namespace = ?"
            params.append(namespace)
        if not hard:
            cur = self.conn.execute(
                "UPDATE memories SET is_active = 0, updated_at = datetime('now') "
                f"WHERE {predicate}",
                params,
            )
            if cur.rowcount:
                self.conn.execute(
                    "UPDATE memory_relations SET is_active = 0, updated_at = datetime('now') "
                    "WHERE (source_id = ? OR target_id = ?) AND is_active = 1",
                    (memory_id, memory_id),
                )
        else:
            self.conn.execute(f"DELETE FROM memories WHERE {predicate}", params)
        self.conn.commit()

    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Archive one memory and preserve the reason in metadata."""
        memory = self.get_memory(memory_id, namespace=namespace)
        if memory is None:
            return
        merged_metadata = dict(memory.metadata)
        merged_metadata["archival_reason"] = reason
        if metadata:
            merged_metadata.update(metadata)
        memory.is_active = False
        memory.metadata = merged_metadata
        self.update_memory(memory, namespace=namespace, commit=commit)
        self.conn.execute(
            "UPDATE memory_relations SET is_active = 0, updated_at = datetime('now') "
            "WHERE (source_id = ? OR target_id = ?) AND is_active = 1",
            (memory_id, memory_id),
        )
        if commit:
            self.conn.commit()

    def count_memories(
        self, *, active_only: bool = True, namespace: str | None = None
    ) -> int:
        """Total number of memories in the store."""
        query = "SELECT COUNT(*) FROM memories"
        predicates: list[str] = []
        params: list[Any] = []
        if active_only:
            predicates.append("is_active = 1")
        if namespace is not None:
            predicates.append("namespace = ?")
            params.append(namespace)
        if predicates:
            query += " WHERE " + " AND ".join(predicates)
        row = self.conn.execute(query, params).fetchone()
        return row[0]


    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def get_all_active_memories(
        self, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Return all active memories, optionally constrained to a namespace."""
        query = "SELECT * FROM memories WHERE is_active = 1"
        params: list[Any] = []
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_type(
        self, memory_type: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Filter active memories by type and optional namespace."""
        query = "SELECT * FROM memories WHERE memory_type = ? AND is_active = 1"
        params: list[Any] = [memory_type]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_tag(
        self, tag: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Find active memories by tag and optional namespace."""
        query = """SELECT m.* FROM memories m
               JOIN memory_tags t ON m.id = t.memory_id
               WHERE t.tag = ? AND m.is_active = 1"""
        params: list[Any] = [tag]
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        query += " ORDER BY m.importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_created_since(
        self, since: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Find active memories created after a timestamp."""
        query = "SELECT * FROM memories WHERE created_at >= ? AND is_active = 1"
        params: list[Any] = [since]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ------------------------------------------------------------------
    # Typed memory relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        relation: MemoryRelation,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        """Persist an edge atomically, returning one stable row id."""
        RelationManager.validate_relation(relation)
        if namespace is not None:
            if relation.namespace is not None and relation.namespace != namespace:
                raise ValueError("relation namespace does not match requested namespace")
            relation.namespace = namespace

        source_row = self.conn.execute(
            "SELECT id, namespace, is_active FROM memories WHERE id = ?",
            (relation.source_id,),
        ).fetchone()
        target_row = self.conn.execute(
            "SELECT id, namespace, is_active FROM memories WHERE id = ?",
            (relation.target_id,),
        ).fetchone()
        if source_row is None or target_row is None:
            raise ValueError("relation endpoints must reference existing memories")
        if not source_row["is_active"] or not target_row["is_active"]:
            raise ValueError("relation endpoints must be active")
        if source_row["namespace"] != target_row["namespace"]:
            raise ValueError("relation endpoints must share a namespace")
        endpoint_namespace = source_row["namespace"]
        if relation.namespace is None:
            relation.namespace = endpoint_namespace
        elif relation.namespace != endpoint_namespace:
            raise ValueError("relation namespace does not match memory endpoints")

        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO memory_relations
                (source_id, target_id, relation_type, confidence, namespace,
                 created_at, updated_at, source, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                confidence = excluded.confidence,
                updated_at = excluded.updated_at,
                source = excluded.source,
                is_active = 1""",
            (
                relation.source_id,
                relation.target_id,
                relation.relation_type,
                relation.confidence,
                relation.namespace,
                relation.created_at or now,
                relation.updated_at or now,
                relation.source,
                1,
            ),
        )
        row = self.conn.execute(
            "SELECT id, created_at, updated_at FROM memory_relations "
            "WHERE source_id = ? AND target_id = ? AND relation_type = ? "
            "AND COALESCE(namespace, '') = COALESCE(?, '')",
            (relation.source_id, relation.target_id, relation.relation_type, relation.namespace),
        ).fetchone()
        if row is None:
            raise RuntimeError("relation insert did not produce a row")
        relation.id = row["id"]
        relation.created_at = row["created_at"]
        relation.updated_at = row["updated_at"]
        relation.is_active = True
        if commit:
            self.conn.commit()
        return int(row["id"])

    def get_relations(
        self,
        source_id: int | None = None,
        *,
        namespace: str | None = None,
        active_only: bool = True,
        target_id: int | None = None,
        relation_type: str | None = None,
    ) -> list[MemoryRelation]:
        """Return edges whose endpoints satisfy namespace/lifecycle filters."""
        for name, value in (("source_id", source_id), ("target_id", target_id)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"relation {name} must be an integer")
        query = (
            "SELECT r.* FROM memory_relations r "
            "JOIN memories s ON s.id = r.source_id "
            "JOIN memories t ON t.id = r.target_id"
        )
        predicates: list[str] = []
        params: list[Any] = []
        if source_id is not None:
            predicates.append("r.source_id = ?")
            params.append(source_id)
        if target_id is not None:
            predicates.append("r.target_id = ?")
            params.append(target_id)
        if namespace is not None:
            predicates.append("r.namespace = ?")
            params.append(namespace)
        if relation_type is not None:
            RelationManager.validate_type(relation_type)
            predicates.append("r.relation_type = ?")
            params.append(relation_type)
        if active_only:
            predicates.extend(("r.is_active = 1", "s.is_active = 1", "t.is_active = 1"))
        if predicates:
            query += " WHERE " + " AND ".join(predicates)
        query += " ORDER BY r.id"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_relation(row) for row in rows]

    def deactivate_relation(
        self,
        relation_id: int,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Deactivate an edge while retaining its audit row."""
        query = "UPDATE memory_relations SET is_active = 0, updated_at = ? WHERE id = ?"
        params: list[Any] = [datetime.now().isoformat(), relation_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        self.conn.execute(query, params)
        if commit:
            self.conn.commit()

    def _row_to_relation(self, row: sqlite3.Row) -> MemoryRelation:
        return MemoryRelation(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            confidence=float(row["confidence"]),
            namespace=row["namespace"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source=row["source"],
            is_active=bool(row["is_active"]),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embedding(
        self,
        memory_id: int,
        vector: bytes,
        model_name: str,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        if namespace is not None:
            exists = self.conn.execute(
                "SELECT 1 FROM memories WHERE id = ? AND namespace = ?",
                (memory_id, namespace),
            ).fetchone()
            if exists is None:
                return
        self.conn.execute(
            """INSERT OR REPLACE INTO embeddings (memory_id, vector, model_name)
               VALUES (?, ?, ?)""",
            (memory_id, vector, model_name),
        )
        if commit:
            self.conn.commit()

    def get_embedding(
        self, memory_id: int, *, namespace: str | None = None
    ) -> tuple[bytes, str] | None:
        """Return (vector_blob, model_name) or None."""
        query = """SELECT e.vector, e.model_name FROM embeddings e
                   JOIN memories m ON e.memory_id = m.id
                   WHERE e.memory_id = ?"""
        params: list[Any] = [memory_id]
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        return (row["vector"], row["model_name"])

    def get_all_embeddings(
        self, *, namespace: str | None = None
    ) -> list[tuple[int, bytes]]:
        """Return (memory_id, vector) for all active memories in a namespace."""
        query = """SELECT e.memory_id, e.vector
               FROM embeddings e
               JOIN memories m ON e.memory_id = m.id
               WHERE m.is_active = 1"""
        params: list[Any] = []
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        rows = self.conn.execute(query, params).fetchall()
        return [(r["memory_id"], r["vector"]) for r in rows]

    def get_embedding_count(self, *, namespace: str | None = None) -> int:
        query = """SELECT COUNT(*) FROM embeddings e
                   JOIN memories m ON e.memory_id = m.id"""
        params: list[Any] = []
        if namespace is not None:
            query += " WHERE m.namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        return row[0]
    def record_access(
        self,
        accesses: list[tuple[int, int]],
        *,
        namespace: str | None = None,
        accessed_at: str | None = None,
        commit: bool = True,
    ) -> None:
        """Persist access timestamps/counts without exposing SQL to adapters."""
        now_iso = accessed_at or datetime.now().isoformat()
        if namespace is None:
            params = [(now_iso, count, memory_id) for memory_id, count in accesses]
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at = ?, access_count = ? "
                "WHERE id = ?",
                params,
            )
        else:
            params = [
                (now_iso, count, memory_id, namespace)
                for memory_id, count in accesses
            ]
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at = ?, access_count = ? "
                "WHERE id = ? AND namespace = ?",
                params,
            )
        if commit:
            self.conn.commit()

    # ------------------------------------------------------------------
    def create_session(
        self,
        label: str = "",
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (label, started_at, namespace) "
            "VALUES (?, datetime('now'), ?)",
            (label or None, namespace),
        )
        if commit:
            self.conn.commit()
        return cur.lastrowid

    def end_session(
        self,
        session_id: int,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        query = "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?"
        params: list[Any] = [session_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        self.conn.execute(query, params)
        if commit:
            self.conn.commit()

    def link_memory_to_session(
        self,
        session_id: int,
        memory_id: int,
        turn_index: int | None = None,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        if namespace is not None:
            valid = self.conn.execute(
                """SELECT 1 FROM sessions s JOIN memories m
                   ON s.namespace = m.namespace
                   WHERE s.id = ? AND m.id = ? AND s.namespace = ?""",
                (session_id, memory_id, namespace),
            ).fetchone()
            if valid is None:
                return
        self.conn.execute(
            """INSERT OR IGNORE INTO session_memories
               (session_id, memory_id, turn_index) VALUES (?, ?, ?)""",
            (session_id, memory_id, turn_index),
        )
        if commit:
            self.conn.commit()

    def get_session_memories(
        self, session_id: int, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        query = """SELECT m.* FROM memories m
               JOIN session_memories sm ON m.id = sm.memory_id
               JOIN sessions s ON s.id = sm.session_id
               WHERE sm.session_id = ?"""
        params: list[Any] = [session_id]
        if namespace is not None:
            query += " AND s.namespace = ? AND m.namespace = ?"
            params.extend([namespace, namespace])
        query += " ORDER BY sm.turn_index ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_recent_sessions(
        self, limit: int = 10, *, namespace: str | None = None
    ) -> list[SessionRecord]:
        query = "SELECT * FROM sessions"
        params: list[Any] = []
        if namespace is not None:
            query += " WHERE namespace = ?"
            params.append(namespace)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            memory_query = """SELECT sm.memory_id FROM session_memories sm
                              JOIN memories m ON m.id = sm.memory_id
                              WHERE sm.session_id = ?"""
            memory_params: list[Any] = [row["id"]]
            if namespace is not None:
                memory_query += " AND m.namespace = ?"
                memory_params.append(namespace)
            mem_rows = self.conn.execute(memory_query, memory_params).fetchall()
            result.append(
                SessionRecord(
                    id=row["id"],
                    label=row["label"] or "",
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    memory_ids=[m["memory_id"] for m in mem_rows],
                    namespace=row["namespace"],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def search_keywords(
        self,
        query: str,
        limit: int = 20,
        *,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """Naive SQL LIKE search, optionally constrained to a namespace."""
        pattern = f"%{query}%"
        sql = """SELECT * FROM memories
               WHERE is_active = 1
                 AND (content LIKE ? OR metadata LIKE ?)"""
        params: list[Any] = [pattern, pattern]
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_strengths(
        self,
        updates: list[tuple[float, int]],
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Batch update memory strengths, optionally constrained to a namespace."""
        now = datetime.now().isoformat()
        query = "UPDATE memories SET strength = ?, updated_at = ? WHERE id = ?"
        if namespace is not None:
            query += " AND namespace = ?"
            params = [(s, now, mid, namespace) for s, mid in updates]
        else:
            params = [(s, now, mid) for s, mid in updates]
        self.conn.executemany(query, params)
        if commit:
            self.conn.commit()

    def archive_below_threshold(
        self,
        threshold: float,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        """Archive memories below a strength threshold in an optional namespace."""
        query = "SELECT id, metadata FROM memories WHERE strength < ? AND is_active = 1"
        params: list[Any] = [threshold]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        rows = self.conn.execute(query, params).fetchall()
        now = datetime.now().isoformat()
        for row in rows:
            raw_metadata = row["metadata"]
            metadata = json.loads(raw_metadata) if raw_metadata else {}
            metadata["archival_reason"] = "decay"
            metadata["archived_at"] = now
            update_query = (
                "UPDATE memories SET is_active = 0, metadata = ?, updated_at = ? "
                "WHERE id = ?"
            )
            update_params: list[Any] = [json.dumps(metadata), now, row["id"]]
            if namespace is not None:
                update_query += " AND namespace = ?"
                update_params.append(namespace)
            self.conn.execute(update_query, update_params)
        if commit:
            self.conn.commit()
        return len(rows)

    def delete_archived_older_than(
        self, days: int = 90, *, namespace: str | None = None
    ) -> int:
        """Hard-delete archived memories older than N days in an optional namespace."""
        query = """DELETE FROM memories
               WHERE is_active = 0
                 AND updated_at < datetime('now', ?)"""
        params: list[Any] = [f"-{days} days"]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        cur = self.conn.execute(query, params)
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        # Tags are in a separate table, always fetch from there
        tags_rows = self.conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY rowid",
            (row["id"],),
        ).fetchall()
        tags = [t["tag"] for t in tags_rows]
        raw_metadata = row["metadata"]
        if raw_metadata is None:
            metadata = {}
        elif isinstance(raw_metadata, str):
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        else:
            metadata = raw_metadata
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=float(row["importance"]),
            strength=float(row["strength"]),
            access_count=int(row["access_count"]),
            last_accessed_at=row["last_accessed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
            tags=tags,
            is_active=bool(row["is_active"]),
            namespace=row["namespace"],
            confidence=(
                float(row["confidence"])
                if row["confidence"] is not None
                else None
            ),
            sensitivity=row["sensitivity"],
            source=row["source"],
            superseded_by=row["superseded_by"],
            last_decision_reason=row["last_decision_reason"],
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")
        self.conn.commit()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
