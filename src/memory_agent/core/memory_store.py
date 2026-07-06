"""SQLite-backed persistent memory storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_agent.models import MemoryRecord, SessionRecord


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
        is_active INTEGER NOT NULL DEFAULT 1
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
        ended_at TEXT
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

    def initialize(self) -> None:
        """Create tables and indexes."""
        self.conn.executescript(self.SCHEMA_SQL)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def add_memory(self, memory: MemoryRecord, *, commit: bool = True) -> int:
        """Insert a new memory record. Returns the new row id."""
        now = datetime.now().isoformat()
        cur = self.conn.execute(
            """INSERT INTO memories
               (content, memory_type, importance, strength,
                access_count, last_accessed_at, created_at, updated_at,
                metadata, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        memory_id = cur.lastrowid
        memory.id = memory_id

        # Insert tags
        for tag in memory.tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )

        if commit:
            self.conn.commit()
        return memory_id

    def get_memory(self, memory_id: int) -> MemoryRecord | None:
        """Fetch a single memory by id."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory(self, memory: MemoryRecord, *, commit: bool = True) -> None:
        """Update an existing memory."""
        now = datetime.now().isoformat()
        self.conn.execute(
            """UPDATE memories SET
               content = ?, memory_type = ?, importance = ?, strength = ?,
               access_count = ?, last_accessed_at = ?, updated_at = ?,
               metadata = ?, is_active = ?
               WHERE id = ?""",
            (
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.strength,
                memory.access_count,
                memory.last_accessed_at or now,
                now,
                json.dumps(memory.metadata),
                1 if memory.is_active else 0,
                memory.id,
            ),
        )
        # Sync tags: delete all, re-insert
        if memory.id is not None:
            self.conn.execute(
                "DELETE FROM memory_tags WHERE memory_id = ?", (memory.id,)
            )
            for tag in memory.tags:
                self.conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (memory.id, tag),
                )
        if commit:
            self.conn.commit()

    def delete_memory(self, memory_id: int, *, hard: bool = False) -> None:
        """Soft-delete (mark inactive) or hard-delete a memory."""
        if hard:
            self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        else:
            self.conn.execute(
                "UPDATE memories SET is_active = 0, updated_at = datetime('now') WHERE id = ?",
                (memory_id,),
            )
        self.conn.commit()

    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> None:
        """Archive one memory and preserve the reason in metadata."""
        memory = self.get_memory(memory_id)
        if memory is None:
            return

        merged_metadata = dict(memory.metadata)
        merged_metadata["archival_reason"] = reason
        if metadata:
            merged_metadata.update(metadata)

        memory.is_active = False
        memory.metadata = merged_metadata
        self.update_memory(memory, commit=commit)

    def count_memories(self, *, active_only: bool = True) -> int:
        """Total number of memories in the store."""
        if active_only:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_active = 1"
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def get_all_active_memories(self) -> list[MemoryRecord]:
        """Return all active (non-archived) memories."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE is_active = 1 ORDER BY importance DESC"
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_type(self, memory_type: str) -> list[MemoryRecord]:
        """Filter memories by type."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE memory_type = ? AND is_active = 1 ORDER BY importance DESC",
            (memory_type,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_tag(self, tag: str) -> list[MemoryRecord]:
        """Find memories tagged with a specific tag."""
        rows = self.conn.execute(
            """SELECT m.* FROM memories m
               JOIN memory_tags t ON m.id = t.memory_id
               WHERE t.tag = ? AND m.is_active = 1
               ORDER BY m.importance DESC""",
            (tag,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_created_since(self, since: str) -> list[MemoryRecord]:
        """Find memories created after a given ISO timestamp."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE created_at >= ? AND is_active = 1 ORDER BY created_at",
            (since,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embedding(
        self, memory_id: int, vector: bytes, model_name: str, *, commit: bool = True
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO embeddings (memory_id, vector, model_name)
               VALUES (?, ?, ?)""",
            (memory_id, vector, model_name),
        )
        if commit:
            self.conn.commit()

    def get_embedding(self, memory_id: int) -> tuple[bytes, str] | None:
        """Return (vector_blob, model_name) or None."""
        row = self.conn.execute(
            "SELECT vector, model_name FROM embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return (row["vector"], row["model_name"])

    def get_all_embeddings(self) -> list[tuple[int, bytes]]:
        """Return (memory_id, vector) for all active memories."""
        rows = self.conn.execute(
            """SELECT e.memory_id, e.vector
               FROM embeddings e
               JOIN memories m ON e.memory_id = m.id
               WHERE m.is_active = 1"""
        ).fetchall()
        return [(r["memory_id"], r["vector"]) for r in rows]

    def get_embedding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, label: str = "", *, commit: bool = True) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (label, started_at) VALUES (?, datetime('now'))",
            (label or None,),
        )
        if commit:
            self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int, *, commit: bool = True) -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        if commit:
            self.conn.commit()

    def link_memory_to_session(
        self, session_id: int, memory_id: int, turn_index: int | None = None, *, commit: bool = True
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO session_memories (session_id, memory_id, turn_index) VALUES (?, ?, ?)",
            (session_id, memory_id, turn_index),
        )
        if commit:
            self.conn.commit()

    def get_session_memories(self, session_id: int) -> list[MemoryRecord]:
        rows = self.conn.execute(
            """SELECT m.* FROM memories m
               JOIN session_memories sm ON m.id = sm.memory_id
               WHERE sm.session_id = ?
               ORDER BY sm.turn_index ASC""",
            (session_id,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_recent_sessions(self, limit: int = 10) -> list[SessionRecord]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            mem_rows = self.conn.execute(
                "SELECT memory_id FROM session_memories WHERE session_id = ?",
                (r["id"],),
            ).fetchall()
            result.append(
                SessionRecord(
                    id=r["id"],
                    label=r["label"] or "",
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    memory_ids=[m["memory_id"] for m in mem_rows],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def search_keywords(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        """Naive SQL LIKE search (fallback when no embeddings)."""
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM memories
               WHERE is_active = 1
                 AND (content LIKE ? OR metadata LIKE ?)
               ORDER BY importance DESC
               LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_strengths(self, updates: list[tuple[float, int]], *, commit: bool = True) -> None:
        """Batch update memory strengths.

        Args:
            updates: list of (new_strength, memory_id) tuples.
        """
        now = datetime.now().isoformat()
        self.conn.executemany(
            "UPDATE memories SET strength = ?, updated_at = ? WHERE id = ?",
            [(s, now, mid) for s, mid in updates],
        )
        if commit:
            self.conn.commit()

    def archive_below_threshold(self, threshold: float, *, commit: bool = True) -> int:
        """Archive memories whose strength has fallen below threshold.

        Returns the number of memories archived.
        """
        rows = self.conn.execute(
            "SELECT id, metadata FROM memories WHERE strength < ? AND is_active = 1",
            (threshold,),
        ).fetchall()
        now = datetime.now().isoformat()

        for row in rows:
            raw_metadata = row["metadata"]
            if raw_metadata:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            else:
                metadata = {}
            metadata["archival_reason"] = "decay"
            metadata["archived_at"] = now
            self.conn.execute(
                """UPDATE memories
                   SET is_active = 0, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (json.dumps(metadata), now, row["id"]),
            )

        if commit:
            self.conn.commit()
        return len(rows)

    def delete_archived_older_than(self, days: int = 90) -> int:
        """Hard-delete archived memories older than N days."""
        cur = self.conn.execute(
            """DELETE FROM memories
               WHERE is_active = 0
                 AND updated_at < datetime('now', ?)""",
            (f"-{days} days",),
        )
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        # Tags are in a separate table, always fetch from there
        tags_rows = self.conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ?", (row["id"],)
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
