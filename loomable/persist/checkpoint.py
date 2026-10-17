"""loomable.persist.checkpoint - Checkpointer protocol and storage providers.

Defines a pluggable ``Checkpointer`` protocol for persisting run state at
boundaries, enabling resume of interrupted runs, resumable streaming, and
durable human-in-the-loop workflows.

Ships two zero-dependency providers (stdlib only):
- ``JsonFileCheckpointer`` (default) — one JSON file per checkpoint in a directory.
  Human-readable, inspectable, debuggable. Best for most workloads.
- ``SQLiteCheckpointer`` — single database file. Better for high-frequency
  checkpointing (e.g. every LLM call).

This module lives at the edge — the kernel imports none of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckpointBusy(RuntimeError):
    """A worker already owns this checkpoint thread."""


@asynccontextmanager
async def _local_claim(root: Path, thread_id: str):
    # Keep the lock inode stable. SQLite releases the transaction on process exit.
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(thread_id.encode()).hexdigest()
    connection = sqlite3.connect(root / name, timeout=0)
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise CheckpointBusy("Checkpoint thread already has an owner") from exc
            raise
        yield
    finally:
        connection.close()


@dataclass
class PendingAction:
    """A proposed action that hasn't been executed yet (durable HITL).

    When a tool call is proposed but awaiting approval, it's stored here so that
    a process restart can resume at the "awaiting confirmation" state rather than
    replaying from scratch.
    """

    tool_name: str
    call_id: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # "pending" | "approved" | "rejected"


@dataclass
class Checkpoint:
    """A snapshot of run state at a boundary.

    Attributes:
        thread_id: Identifier for the logical conversation/thread.
        step: The step counter at checkpoint time.
        session_state: Serializable session state (mirrors SessionStore payload).
        stream_text: Accumulated output text (for resumable streaming).
        usage: Token usage at checkpoint time.
        complete: Whether this represents a completed run (True) or partial (False).
        pending: Proposed actions awaiting approval (durable HITL).
        event_kind: The event that triggered this checkpoint (for event-driven triggers).
        lineage_id: Lineage identifier for fork support.
        timestamp: Unix timestamp of checkpoint creation.
    """

    thread_id: str
    step: int
    session_state: dict[str, Any] = field(default_factory=dict)
    stream_text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    complete: bool = True
    pending: list[PendingAction] = field(default_factory=list)
    event_kind: str = ""
    lineage_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "thread_id": self.thread_id,
            "step": self.step,
            "session_state": self.session_state,
            "stream_text": self.stream_text,
            "usage": self.usage,
            "complete": self.complete,
            "pending": [
                {"tool_name": p.tool_name, "call_id": p.call_id,
                 "args": p.args, "status": p.status}
                for p in self.pending
            ],
            "event_kind": self.event_kind,
            "lineage_id": self.lineage_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Deserialize from a dict."""
        pending = [
            PendingAction(
                tool_name=p["tool_name"],
                call_id=p["call_id"],
                args=p.get("args", {}),
                status=p.get("status", "pending"),
            )
            for p in data.get("pending", [])
        ]
        return cls(
            thread_id=data["thread_id"],
            step=data["step"],
            session_state=data.get("session_state", {}),
            stream_text=data.get("stream_text", ""),
            usage=data.get("usage", {}),
            complete=data.get("complete", True),
            pending=pending,
            event_kind=data.get("event_kind", ""),
            lineage_id=data.get("lineage_id", ""),
            timestamp=data.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Checkpointer(Protocol):
    """Protocol for pluggable checkpoint stores.

    Implementations must support put, get (latest), list (commit order),
    and optionally fork and prune for advanced workflows.
    """

    async def put(self, cp: Checkpoint) -> None:
        """Persist a checkpoint under its thread_id."""
        ...

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Retrieve the latest checkpoint for a thread, or None."""
        ...

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread in commit order."""
        ...


# ---------------------------------------------------------------------------
# Event-driven checkpoint trigger config
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Configuration for event-driven checkpoint triggers.

    Attributes:
        on_events: Event kinds that trigger a checkpoint write.
            Defaults to ["run_end"] (one checkpoint per completed run).
            Use ["*"] to checkpoint on every event.
        max_checkpoints: Maximum checkpoints to retain per thread.
            Oldest are pruned after each write. None = no limit.
        location: Storage location (directory for JsonFile, path for SQLite).
    """

    on_events: list[str] = field(default_factory=lambda: ["run_end"])
    max_checkpoints: int | None = None
    location: str = ".checkpoints"


# ---------------------------------------------------------------------------
# JsonFileCheckpointer (default — human-readable, one file per checkpoint)
# ---------------------------------------------------------------------------


class JsonFileCheckpointer:
    """A file-based Checkpointer — one JSON file per checkpoint.

    Human-readable, inspectable with any text editor, easy to debug.
    Files are named ``{timestamp}_{uuid}.json`` inside a directory per thread.

    This is the recommended default for most workloads (run-level or
    task-level checkpoint frequency).
    """

    def __init__(self, location: str = ".checkpoints", max_checkpoints: int | None = None) -> None:
        self._location = location
        self._max_checkpoints = max_checkpoints
        os.makedirs(location, exist_ok=True)

    def _thread_path(self, thread_id: str) -> str:
        if not thread_id or thread_id in {".", ".."} or any(
            separator in thread_id for separator in ("/", chr(92), chr(0))
        ) or os.path.splitdrive(thread_id)[0]:
            raise ValueError("Checkpoint thread ID must be a single path component")
        path = os.path.join(self._location, thread_id)
        if os.path.islink(path):
            raise ValueError("Checkpoint directory must not be a symbolic link")
        return path

    def _thread_dir(self, thread_id: str) -> str:
        path = self._thread_path(thread_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _checkpoint_filename(self, cp: Checkpoint) -> str:
        """Generate a filename for a checkpoint: {timestamp}_{uuid}.json"""
        ts = f"{cp.timestamp:.6f}".replace(".", "_")
        uid = uuid.uuid4().hex[:8]
        return f"{ts}_{uid}.json"

    def claim(self, thread_id: str):
        return _local_claim(Path(self._location).resolve() / ".ownership", thread_id)

    async def put(self, cp: Checkpoint) -> None:
        """Write a checkpoint as a JSON file.

        Always refreshes ``cp.timestamp`` so ``get()`` (latest-by-filename)
        returns this write even when the caller mutated an older checkpoint
        in place (e.g. Workflow.approve).
        """
        cp.timestamp = time.time()
        thread_dir = self._thread_dir(cp.thread_id)
        filename = self._checkpoint_filename(cp)
        filepath = os.path.join(thread_dir, filename)

        fd, temporary = tempfile.mkstemp(dir=thread_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cp.to_dict(), f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, filepath)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        # Prune if max_checkpoints is set
        if self._max_checkpoints is not None:
            await self._prune(cp.thread_id)

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread (by filename sort order)."""
        thread_dir = self._thread_path(thread_id)
        if not os.path.isdir(thread_dir):
            return None

        files = sorted(
            [f for f in os.listdir(thread_dir) if f.endswith(".json")],
            reverse=True,
        )
        if not files:
            return None

        filepath = os.path.join(thread_dir, files[0])
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Checkpoint.from_dict(data)

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread in commit order (oldest first)."""
        thread_dir = self._thread_path(thread_id)
        if not os.path.isdir(thread_dir):
            return []

        files = sorted(
            [f for f in os.listdir(thread_dir) if f.endswith(".json")]
        )
        checkpoints: list[Checkpoint] = []
        for filename in files:
            filepath = os.path.join(thread_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoints.append(Checkpoint.from_dict(data))
        return checkpoints

    async def fork(self, source_thread_id: str, new_thread_id: str) -> Checkpoint | None:
        """Fork from the latest checkpoint of source into a new thread.

        Copies the checkpoint under a new thread_id and lineage, so both
        can continue independently.
        """
        latest = await self.get(source_thread_id)
        if latest is None:
            return None

        forked = Checkpoint(
            thread_id=new_thread_id,
            step=latest.step,
            session_state=latest.session_state.copy(),
            stream_text=latest.stream_text,
            usage=latest.usage.copy(),
            complete=latest.complete,
            pending=[PendingAction(**p.__dict__) for p in latest.pending],
            event_kind="fork",
            lineage_id=new_thread_id,
            timestamp=time.time(),
        )
        await self.put(forked)
        return forked

    async def _prune(self, thread_id: str) -> None:
        """Remove oldest checkpoints beyond the max_checkpoints limit."""
        if self._max_checkpoints is None:
            return

        thread_dir = self._thread_path(thread_id)
        if not os.path.isdir(thread_dir):
            return

        files = sorted(
            [f for f in os.listdir(thread_dir) if f.endswith(".json")]
        )
        excess = len(files) - self._max_checkpoints
        if excess > 0:
            for filename in files[:excess]:
                os.remove(os.path.join(thread_dir, filename))


# ---------------------------------------------------------------------------
# SQLiteCheckpointer (high-frequency alternative)
# ---------------------------------------------------------------------------


class InMemoryCheckpointer:
    """An in-memory Checkpointer for testing only.

    Stores checkpoints in a plain dict — all data is lost on process exit.
    Use JsonFileCheckpointer or SQLiteCheckpointer for durable persistence.

    This class exists so that tests can exercise checkpoint/resume logic
    without touching the filesystem or SQLite.
    """

    def __init__(self, max_checkpoints: int | None = None) -> None:
        self._store: dict[str, list[Checkpoint]] = {}
        self._claims: set[str] = set()
        self._max_checkpoints = max_checkpoints

    @asynccontextmanager
    async def claim(self, thread_id: str):
        if thread_id in self._claims:
            raise CheckpointBusy("Checkpoint thread already has an owner")
        self._claims.add(thread_id)
        try:
            yield
        finally:
            self._claims.remove(thread_id)

    async def put(self, cp: Checkpoint) -> None:
        """Store a checkpoint in memory."""
        thread_list = self._store.setdefault(cp.thread_id, [])
        thread_list.append(cp)

        if self._max_checkpoints is not None:
            excess = len(thread_list) - self._max_checkpoints
            if excess > 0:
                self._store[cp.thread_id] = thread_list[excess:]

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread."""
        thread_list = self._store.get(thread_id)
        if not thread_list:
            return None
        return thread_list[-1]

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread in commit order."""
        return list(self._store.get(thread_id, []))

    async def fork(self, source_thread_id: str, new_thread_id: str) -> Checkpoint | None:
        """Fork from the latest checkpoint of source into a new thread."""
        latest = await self.get(source_thread_id)
        if latest is None:
            return None
        forked = Checkpoint(
            thread_id=new_thread_id,
            step=latest.step,
            session_state=dict(latest.session_state),
            stream_text=latest.stream_text,
            usage=dict(latest.usage),
            complete=latest.complete,
            pending=[
                PendingAction(
                    tool_name=p.tool_name,
                    call_id=p.call_id,
                    args=dict(p.args),
                    status=p.status,
                )
                for p in latest.pending
            ],
            event_kind="fork",
            lineage_id=new_thread_id,
            timestamp=time.time(),
        )
        await self.put(forked)
        return forked


class SQLiteCheckpointer:
    """A Checkpointer backed by stdlib sqlite3.

    Better for high-frequency checkpointing (every LLM call, every tool call)
    where file-per-checkpoint would create I/O overhead. Uses WAL journaling
    for concurrent read/write performance.

    Pass a file path for durable persistence, or ":memory:" for testing.
    """

    def __init__(self, db_path: str = ":memory:", max_checkpoints: int | None = None) -> None:
        self._claims: set[str] = set()
        self._lock_root = None if str(db_path) in {":memory:", ""} else Path(
            str(Path(db_path).resolve()) + ".ownership"
        )
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._max_checkpoints = max_checkpoints
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                payload TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
            ON checkpoints(thread_id, id)
        """)
        self._conn.commit()

    @asynccontextmanager
    async def claim(self, thread_id: str):
        if self._lock_root is None:
            if thread_id in self._claims:
                raise CheckpointBusy("Checkpoint thread already has an owner")
            self._claims.add(thread_id)
            try:
                yield
            finally:
                self._claims.remove(thread_id)
        else:
            async with _local_claim(self._lock_root, thread_id):
                yield

    async def put(self, cp: Checkpoint) -> None:
        """Persist a checkpoint."""
        self._conn.execute(
            "INSERT INTO checkpoints (thread_id, step, payload, timestamp) VALUES (?, ?, ?, ?)",
            (cp.thread_id, cp.step, json.dumps(cp.to_dict()), cp.timestamp),
        )
        self._conn.commit()

        if self._max_checkpoints is not None:
            await self._prune(cp.thread_id)

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread."""
        row = self._conn.execute(
            "SELECT payload FROM checkpoints WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint.from_dict(json.loads(row[0]))

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread in commit order."""
        rows = self._conn.execute(
            "SELECT payload FROM checkpoints WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
        return [Checkpoint.from_dict(json.loads(row[0])) for row in rows]

    async def fork(self, source_thread_id: str, new_thread_id: str) -> Checkpoint | None:
        """Fork from the latest checkpoint into a new thread."""
        latest = await self.get(source_thread_id)
        if latest is None:
            return None

        forked = Checkpoint(
            thread_id=new_thread_id,
            step=latest.step,
            session_state=latest.session_state.copy(),
            stream_text=latest.stream_text,
            usage=latest.usage.copy(),
            complete=latest.complete,
            pending=[PendingAction(**p.__dict__) for p in latest.pending],
            event_kind="fork",
            lineage_id=new_thread_id,
            timestamp=time.time(),
        )
        await self.put(forked)
        return forked

    async def _prune(self, thread_id: str) -> None:
        """Remove oldest checkpoints beyond the limit."""
        if self._max_checkpoints is None:
            return
        count = self._conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]
        excess = count - self._max_checkpoints
        if excess > 0:
            self._conn.execute(
                """DELETE FROM checkpoints WHERE id IN (
                    SELECT id FROM checkpoints WHERE thread_id = ?
                    ORDER BY id ASC LIMIT ?
                )""",
                (thread_id, excess),
            )
            self._conn.commit()
