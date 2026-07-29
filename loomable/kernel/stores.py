"""loomable.kernel.stores - Short-term and long-term memory stores.

Provides ShortTermStore (pluggable RDBMS backend, SQLite default),
SessionStore (SQLite-backed session persistence), and related store
implementations.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from loomable.kernel.contracts import MemoryBackend
from loomable.kernel.errors import MemoryBackendError, SessionNotFoundError
from loomable.kernel.models import Session, StructuredSummary, Turn


# ---------------------------------------------------------------------------
# Concrete SQLite backend satisfying the MemoryBackend protocol
# ---------------------------------------------------------------------------


class SQLiteMemoryBackend:
    """SQLite-based implementation of the MemoryBackend protocol.

    Values are stored as JSON-serialized text in a single key-value table.
    This is the default backend for ShortTermStore.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._backend_id = f"sqlite:{db_path}"
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv_store ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL"
                ")"
            )
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise MemoryBackendError(self._backend_id) from exc

    @property
    def backend_id(self) -> str:
        """Identifier for this backend instance."""
        return self._backend_id

    async def read(self, key: str) -> Any:
        """Read a value by key from SQLite."""
        try:
            cursor = self._conn.execute(
                "SELECT value FROM kv_store WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise MemoryBackendError(self._backend_id) from exc
        if row is None:
            return None
        return json.loads(row[0])

    async def write(self, key: str, value: Any) -> None:
        """Write a value by key to SQLite (upsert)."""
        try:
            serialized = json.dumps(value)
            self._conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                (key, serialized),
            )
            self._conn.commit()
        except (sqlite3.Error, TypeError) as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def delete(self, key: str) -> None:
        """Delete a value by key from SQLite."""
        try:
            self._conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def exists(self, key: str) -> bool:
        """Check whether a key exists in SQLite."""
        try:
            cursor = self._conn.execute(
                "SELECT 1 FROM kv_store WHERE key = ?", (key,)
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as exc:
            raise MemoryBackendError(self._backend_id) from exc

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# ShortTermStore - pluggable RDBMS backend (SQLite default)
# ---------------------------------------------------------------------------


class ShortTermStore:
    """Short-term memory store for recent conversational/session state.

    Uses a pluggable MemoryBackend, defaulting to SQLiteMemoryBackend.
    Alternative backends require no agent changes — just supply any object
    satisfying the MemoryBackend protocol.
    """

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        if backend is None:
            backend = SQLiteMemoryBackend()
        self._backend: MemoryBackend = backend

    @property
    def backend(self) -> MemoryBackend:
        """The active memory backend."""
        return self._backend

    async def read(self, key: str) -> Any:
        """Read persisted state by key.

        Returns the persisted value, or None if the key does not exist.
        Raises MemoryBackendError if the backend is unavailable.
        """
        return await self._backend.read(key)

    async def write(self, key: str, value: Any) -> None:
        """Persist state by key.

        Raises MemoryBackendError if the backend is unavailable.
        """
        await self._backend.write(key, value)

    async def delete(self, key: str) -> None:
        """Delete persisted state by key.

        Raises MemoryBackendError if the backend is unavailable.
        """
        await self._backend.delete(key)

    async def exists(self, key: str) -> bool:
        """Check whether a key exists in the store.

        Raises MemoryBackendError if the backend is unavailable.
        """
        return await self._backend.exists(key)


# ---------------------------------------------------------------------------
# SessionStore - SQLite-backed session persistence (out of the box)
# ---------------------------------------------------------------------------


class SessionStore:
    """SQLite-backed session persistence.

    Persists agent sessions using SQLite by default without additional
    configuration. Sessions are saved and resumed by session identifier.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "  session_id TEXT PRIMARY KEY,"
            "  data TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def save(self, session: Session) -> None:
        """Persist the session state.

        Serializes the full session (l1 turns, l2 summaries, step counter,
        and config ref) to SQLite keyed by session_id.
        """
        data = self._serialize_session(session)
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
            (session.session_id, json.dumps(data)),
        )
        self._conn.commit()

    def resume(self, session_id: str) -> Session:
        """Restore the persisted session state by session identifier.

        Raises SessionNotFoundError if the requested session_id does not exist.
        """
        cursor = self._conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        data = json.loads(row[0])
        return self._deserialize_session(data)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_session(session: Session) -> dict[str, Any]:
        """Convert a Session to a JSON-serializable dict."""
        return {
            "session_id": session.session_id,
            "agent_config_ref": session.agent_config_ref,
            "step": session.step,
            "l1": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tokens": t.tokens,
                    "step": t.step,
                }
                for t in session.l1
            ],
            "l2": [
                {
                    "covers_steps_start": s.covers_steps.start,
                    "covers_steps_stop": s.covers_steps.stop,
                    "objectives": s.objectives,
                    "decisions": s.decisions,
                    "text": s.text,
                    "tokens": s.tokens,
                }
                for s in session.l2
            ],
        }

    @staticmethod
    def _deserialize_session(data: dict[str, Any]) -> Session:
        """Reconstruct a Session from a deserialized dict."""
        l1 = [
            Turn(
                role=t["role"],
                content=t["content"],
                tokens=t["tokens"],
                step=t["step"],
            )
            for t in data["l1"]
        ]
        l2 = [
            StructuredSummary(
                covers_steps=range(s["covers_steps_start"], s["covers_steps_stop"]),
                objectives=s["objectives"],
                decisions=s["decisions"],
                text=s["text"],
                tokens=s["tokens"],
            )
            for s in data["l2"]
        ]
        return Session(
            session_id=data["session_id"],
            agent_config_ref=data["agent_config_ref"],
            l1=l1,
            l2=l2,
            step=data["step"],
        )
