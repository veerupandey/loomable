"""PostgreSQL checkpointer (asyncpg).

Production-durable Workflow/Case resume. Optional dependency::

    pip install 'loomable[postgres]'  # or: pip install asyncpg

Tables are created automatically on first use.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from loomable.persist.checkpoint import Checkpoint, PendingAction

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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
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
                CREATE INDEX IF NOT EXISTS {self._table}_thread_ts_idx
                ON {self._fq_table} (thread_id, timestamp DESC, id DESC)
                """
            )
        self._ready = True

    async def _ensure_ready(self) -> Any:
        pool = await self._ensure_pool()
        if not self._ready:
            await self.setup()
        return pool

    async def put(self, cp: Checkpoint) -> None:
        """Persist a checkpoint under its ``thread_id``."""
        pool = await self._ensure_ready()
        cp.timestamp = time.time()
        payload = cp.to_dict()
        async with pool.acquire() as conn:
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
                        ORDER BY timestamp DESC, id DESC
                        OFFSET $2
                    )
                    """,
                    cp.thread_id,
                    int(self._max_checkpoints),
                )

    async def get(self, thread_id: str) -> Checkpoint | None:
        """Retrieve the latest checkpoint for a thread, or None."""
        pool = await self._ensure_ready()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT payload FROM {self._fq_table}
                WHERE thread_id = $1
                ORDER BY timestamp DESC, id DESC
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
                ORDER BY timestamp ASC, id ASC
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
