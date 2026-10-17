"""PostgreSQL checkpointer (asyncpg).

    pip install 'loomable[postgres]'
    docker compose up -d
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from loomable.persist.checkpoint import Checkpoint, CheckpointBusy, PendingAction

__all__ = ["PostgresCheckpointer"]


def _require_asyncpg():
    try:
        import asyncpg  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PostgresCheckpointer requires asyncpg. "
            "Install with: pip install 'loomable[postgres]' or pip install asyncpg"
        ) from exc
    return asyncpg


class PostgresCheckpointer:
    """Async PostgreSQL :class:`~loomable.persist.checkpoint.Checkpointer`.

    Parameters
    ----------
    url:
        ``postgresql://`` / ``postgres://`` DSN, or an existing asyncpg pool/connection.
    table:
        Table name for checkpoint rows (created if missing).
    max_checkpoints:
        Optional prune limit per ``thread_id`` (oldest removed after each put).
    schema:
        Optional Postgres schema (default ``public``).
    """

    def __init__(
        self,
        url: str | Any,
        *,
        table: str = "loomable_checkpoints",
        max_checkpoints: int | None = None,
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._url = url
        self._table = table
        self._schema = schema
        self._max_checkpoints = max_checkpoints
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None
        self._owns_pool = False
        self._ready = False
        self._setup_lock = asyncio.Lock()
        self._claimed_connection: ContextVar = ContextVar("checkpoint_connection", default=None)
        # Allow injecting a pool/connection for tests
        if not isinstance(url, str):
            self._pool = url
            self._owns_pool = False

    @property
    def _fq_table(self) -> str:
        return f'"{self._schema}"."{self._table}"'

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        asyncpg = _require_asyncpg()
        self._pool = await asyncpg.create_pool(
            dsn=str(self._url),
            min_size=self._min_size,
            max_size=self._max_size,
        )
        self._owns_pool = True
        return self._pool

    async def setup(self) -> None:
        """Create schema objects if missing (idempotent)."""
        async with self._setup_lock:
            if self._ready:
                return
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock($1::bigint)",
                        self._lock_id("schema:initialize"),
                    )
                    await conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self._fq_table} (
                            id BIGSERIAL PRIMARY KEY,
                            thread_id TEXT NOT NULL,
                            step INTEGER NOT NULL,
                            payload JSONB NOT NULL,
                            timestamp DOUBLE PRECISION NOT NULL,
                            complete BOOLEAN NOT NULL DEFAULT FALSE
                        )
                        """
                    )
                    await conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self._table}_thread_id_idx
                        ON {self._fq_table} (thread_id, id DESC)
                        """
                    )
                    await conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self._table}_completed_idx
                        ON {self._fq_table} (timestamp, id)
                        WHERE complete AND thread_id LIKE 'agent-session-%'
                        """
                    )
            self._ready = True

    async def _ensure_ready(self) -> Any:
        if not self._ready:
            await self.setup()
        return self._pool

    def _lock_id(self, key: str) -> int:
        lock_name = f"{self._schema}.{self._table}:{key}"
        digest = hashlib.sha256(lock_name.encode()).digest()
        return int.from_bytes(digest[:8], "big", signed=True)

    @asynccontextmanager
    async def claim(self, thread_id: str):
        pool = await self._ensure_ready()
        lock_id = self._lock_id(thread_id)
        async with pool.acquire() as conn:
            if not await conn.fetchval("SELECT pg_try_advisory_lock($1::bigint)", lock_id):
                raise CheckpointBusy("Checkpoint thread already has an owner")
            token = self._claimed_connection.set((thread_id, conn))
            try:
                yield
            finally:
                self._claimed_connection.reset(token)
                # Pool release resets advisory locks even if explicit unlock fails.
                if not conn.is_closed():
                    await conn.fetchval("SELECT pg_advisory_unlock($1::bigint)", lock_id)

    @asynccontextmanager
    async def _connection(self, thread_id: str):
        claimed = self._claimed_connection.get()
        if claimed is not None and claimed[0] == thread_id:
            yield claimed[1]
        else:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                yield conn

    async def put(self, cp: Checkpoint) -> None:
        """Persist a checkpoint under its ``thread_id``."""
        cp.timestamp = time.time()
        payload = cp.to_dict()
        async with self._connection(cp.thread_id) as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._fq_table}
                    (thread_id, step, payload, timestamp, complete)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                """,
                cp.thread_id,
                int(cp.step),
                json.dumps(payload, default=str),
                float(cp.timestamp),
                bool(cp.complete),
            )
            if self._max_checkpoints is not None:
                await conn.execute(
                    f"""
                    DELETE FROM {self._fq_table}
                    WHERE id IN (
                        SELECT id FROM {self._fq_table}
                        WHERE thread_id = $1
                        ORDER BY id DESC
                        OFFSET $2
                    )
                    """,
                    cp.thread_id,
                    int(self._max_checkpoints),
                )

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Retrieve the latest checkpoint for a thread, or None."""
        async with self._connection(thread_id) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT payload FROM {self._fq_table}
                WHERE thread_id = $1
                ORDER BY id DESC
                LIMIT 1
                """,
                thread_id,
            )
        if row is None:
            return None
        data = row["payload"]
        if isinstance(data, str):
            data = json.loads(data)
        return Checkpoint.from_dict(dict(data))

    async def list(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread in commit order (oldest first)."""
        pool = await self._ensure_ready()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT payload FROM {self._fq_table}
                WHERE thread_id = $1
                ORDER BY id ASC
                """,
                thread_id,
            )
        out: list[Checkpoint] = []
        for row in rows:
            data = row["payload"]
            if isinstance(data, str):
                data = json.loads(data)
            out.append(Checkpoint.from_dict(dict(data)))
        return out

    async def prune_completed_agent_sessions(self, older_than_seconds: float, *, limit: int = 100) -> int:
        """Remove up to `limit` expired agent sessions whose latest checkpoint is complete."""
        if not math.isfinite(older_than_seconds) or older_than_seconds < 0 or limit < 1:
            raise ValueError("Retention requires a finite nonnegative age and a positive batch size")
        cutoff = time.time() - older_than_seconds
        pool = await self._ensure_ready()
        async with pool.acquire() as conn:
            candidates = await conn.fetch(
                f"""
                SELECT cp.thread_id, cp.id FROM {self._fq_table} AS cp
                WHERE cp.complete AND cp.timestamp <= $1 AND cp.thread_id LIKE 'agent-session-%'
                  AND NOT EXISTS (
                    SELECT 1 FROM {self._fq_table} AS newer
                    WHERE newer.thread_id = cp.thread_id AND newer.id > cp.id
                  )
                ORDER BY cp.timestamp, cp.id
                LIMIT $2
                """,
                cutoff, limit,
            )
        removed = 0
        for row in candidates:
            thread_id = row["thread_id"]
            try:
                async with self.claim(thread_id), self._connection(thread_id) as conn:
                    # A writer may have acquired and released the claim after the scan.
                    status = await conn.execute(
                        f"""
                        DELETE FROM {self._fq_table} WHERE thread_id = $1 AND $2 = (
                            SELECT id FROM {self._fq_table} WHERE thread_id = $1
                            ORDER BY id DESC LIMIT 1
                        )
                        """,
                        thread_id, row["id"],
                    )
                    removed += status != "DELETE 0"
            except CheckpointBusy:
                continue
        return removed

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
            pending=[PendingAction(**p.__dict__) for p in latest.pending],
            event_kind="fork",
            lineage_id=new_thread_id,
            timestamp=time.time(),
        )
        await self.put(forked)
        return forked

    async def aclose(self) -> None:
        """Close the owned connection pool (no-op for injected pools)."""
        if self._owns_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._owns_pool = False
            self._ready = False

    async def close(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()
