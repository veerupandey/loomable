"""loomable.kernel.stores - Short-term KV and session persistence.

- :class:`ShortTermStore` — pluggable :class:`MemoryBackend` (SQLite default)
- :class:`SessionStore` — SQLite L1/L2 conversation persistence (default)
- :class:`FileSessionStore` — one JSON file per session
- :class:`BackendSessionStore` — any :class:`MemoryBackend` (Postgres, custom, …)
- :class:`InMemoryMemoryBackend` — process-local KV for tests / demos
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loomable.kernel.contracts import MemoryBackend
from loomable.kernel.errors import MemoryBackendError, SessionNotFoundError
from loomable.kernel.models import Session, StructuredSummary, Turn

__all__ = [
    "SQLiteMemoryBackend",
    "InMemoryMemoryBackend",
    "ShortTermStore",
    "SessionStoreProtocol",
    "SessionStore",
    "FileSessionStore",
    "BackendSessionStore",
    "serialize_session",
    "deserialize_session",
]


# ---------------------------------------------------------------------------
# Memory backends
# ---------------------------------------------------------------------------


class SQLiteMemoryBackend:
    """SQLite-based implementation of the MemoryBackend protocol."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._backend_id = f"sqlite:{db_path}"
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
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
        return self._backend_id

    async def read(self, key: str) -> Any:
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
        try:
            self._conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def exists(self, key: str) -> bool:
        try:
            cursor = self._conn.execute(
                "SELECT 1 FROM kv_store WHERE key = ?", (key,)
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as exc:
            raise MemoryBackendError(self._backend_id) from exc

    def read_sync(self, key: str) -> Any:
        cursor = self._conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def write_sync(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class InMemoryMemoryBackend:
    """Process-local MemoryBackend (tests / demos)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._backend_id = "memory"

    @property
    def backend_id(self) -> str:
        return self._backend_id

    async def read(self, key: str) -> Any:
        return self._data.get(key)

    async def write(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data

    def read_sync(self, key: str) -> Any:
        return self._data.get(key)

    def write_sync(self, key: str, value: Any) -> None:
        self._data[key] = value


class ShortTermStore:
    """Short-term KV store over a pluggable MemoryBackend."""

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        if backend is None:
            backend = SQLiteMemoryBackend()
        self._backend: MemoryBackend = backend

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    async def read(self, key: str) -> Any:
        return await self._backend.read(key)

    async def write(self, key: str, value: Any) -> None:
        await self._backend.write(key, value)

    async def delete(self, key: str) -> None:
        await self._backend.delete(key)

    async def exists(self, key: str) -> bool:
        return await self._backend.exists(key)


# ---------------------------------------------------------------------------
# Session serialization (shared by all SessionStore backends)
# ---------------------------------------------------------------------------


def serialize_session(session: Session) -> dict[str, Any]:
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


def deserialize_session(data: dict[str, Any]) -> Session:
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


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Pluggable L1/L2 conversation persistence for :class:`~loomable.agent.Agent`."""

    def save(self, session: Session) -> None: ...

    def resume(self, session_id: str) -> Session: ...


def _run_sync(fn_or_coro: Any) -> Any:
    """Run ``fn()`` or a coroutine off the current event loop if needed."""

    def _call() -> Any:
        if asyncio.iscoroutine(fn_or_coro):
            return asyncio.run(fn_or_coro)
        return fn_or_coro()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _call()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_call).result()


# ---------------------------------------------------------------------------
# Session stores
# ---------------------------------------------------------------------------


class SessionStore:
    """SQLite-backed session persistence (default Agent L1/L2 store)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "  session_id TEXT PRIMARY KEY,"
            "  data TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def save(self, session: Session) -> None:
        data = serialize_session(session)
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
            (session.session_id, json.dumps(data)),
        )
        self._conn.commit()

    def resume(self, session_id: str) -> Session:
        cursor = self._conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return deserialize_session(json.loads(row[0]))

    def close(self) -> None:
        self._conn.close()


class FileSessionStore:
    """One JSON file per session under ``root`` (``{session_id}.json``)."""

    def __init__(self, root: str | Path = ".sessions") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._root / f"{safe}.json"

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(serialize_session(session), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def resume(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.is_file():
            raise SessionNotFoundError(session_id)
        return deserialize_session(json.loads(path.read_text(encoding="utf-8")))


class BackendSessionStore:
    """SessionStore adapter over any :class:`MemoryBackend`.

    Prefer backends that expose ``read_sync`` / ``write_sync`` (thread-safe from
    Agent's async run path). Otherwise falls back to running the async methods
    in a worker thread with a fresh event loop.
    """

    def __init__(self, backend: MemoryBackend, *, key_prefix: str = "session:") -> None:
        self._backend = backend
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def save(self, session: Session) -> None:
        key = self._key(session.session_id)
        payload = serialize_session(session)
        write_sync = getattr(self._backend, "write_sync", None)
        if callable(write_sync):
            _run_sync(lambda: write_sync(key, payload))
            return
        _run_sync(self._backend.write(key, payload))

    def resume(self, session_id: str) -> Session:
        key = self._key(session_id)
        read_sync = getattr(self._backend, "read_sync", None)
        if callable(read_sync):
            data = _run_sync(lambda: read_sync(key))
        else:
            data = _run_sync(self._backend.read(key))
        if data is None:
            raise SessionNotFoundError(session_id)
        if isinstance(data, str):
            data = json.loads(data)
        return deserialize_session(dict(data))
