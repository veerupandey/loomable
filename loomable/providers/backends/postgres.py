"""PostgreSQL MemoryBackend + VectorBackend (asyncpg).

    pip install 'loomable[postgres]'
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from loomable.kernel.errors import MemoryBackendError

__all__ = ["PostgresMemoryBackend", "PgVectorBackend"]


def _require_asyncpg():
    try:
        import asyncpg  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Postgres memory backends require asyncpg. "
            "Install with: pip install 'loomable[postgres]' or pip install asyncpg"
        ) from exc
    return asyncpg


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class _PoolMixin:
    def __init__(
        self,
        url: str | Any,
        *,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._url = url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None
        self._owns_pool = False
        self._ready = False
        if not isinstance(url, str):
            self._pool = url
            self._owns_pool = False

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

    async def aclose(self) -> None:
        if self._owns_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._owns_pool = False
            self._ready = False

    async def close(self) -> None:
        await self.aclose()


class PostgresMemoryBackend(_PoolMixin):
    """Key/value short-term memory backed by PostgreSQL JSONB.

    Compatible with :class:`~loomable.kernel.stores.ShortTermStore`.
    Keys are scoped by ``user_id`` when provided.
    """

    def __init__(
        self,
        url: str | Any,
        *,
        user_id: str | None = None,
        table: str = "loomable_kv",
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        super().__init__(url, min_size=min_size, max_size=max_size)
        self._user_id = user_id or ""
        self._table = table
        self._schema = schema
        self._backend_id = f"postgres:{schema}.{table}:{self._user_id or '_'}"

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def _fq_table(self) -> str:
        return f'"{self._schema}"."{self._table}"'

    async def setup(self) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._fq_table} (
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scope, key)
                )
                """
            )
        self._ready = True

    async def _ensure_ready(self) -> Any:
        pool = await self._ensure_pool()
        if not self._ready:
            await self.setup()
        return pool

    async def read(self, key: str) -> Any:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT value FROM {self._fq_table}
                    WHERE scope = $1 AND key = $2
                    """,
                    self._user_id,
                    key,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc
        if row is None:
            return None
        value = row["value"]
        if isinstance(value, str):
            return json.loads(value)
        return value

    async def write(self, key: str, value: Any) -> None:
        try:
            pool = await self._ensure_ready()
            payload = json.dumps(value, default=str)
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._fq_table} (scope, key, value)
                    VALUES ($1, $2, $3::jsonb)
                    ON CONFLICT (scope, key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    self._user_id,
                    key,
                    payload,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    async def delete(self, key: str) -> None:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    DELETE FROM {self._fq_table}
                    WHERE scope = $1 AND key = $2
                    """,
                    self._user_id,
                    key,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    async def exists(self, key: str) -> bool:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT 1 FROM {self._fq_table}
                    WHERE scope = $1 AND key = $2
                    """,
                    self._user_id,
                    key,
                )
            return row is not None
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    def _oneshot_sync(self, fn: Any) -> Any:
        """Run asyncpg work on a fresh connection+loop (safe from Agent persist)."""
        if not isinstance(self._url, str):
            raise RuntimeError("oneshot requires a DSN string")
        asyncpg = _require_asyncpg()

        async def _runner() -> Any:
            conn = await asyncpg.connect(str(self._url))
            try:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._fq_table} (
                        scope TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (scope, key)
                    )
                    """
                )
                return await fn(conn)
            finally:
                await conn.close()

        return asyncio.run(_runner())

    def read_sync(self, key: str) -> Any:
        if not isinstance(self._url, str):
            from loomable.kernel.stores import _run_sync

            return _run_sync(self.read(key))

        async def _op(conn: Any) -> Any:
            row = await conn.fetchrow(
                f"""
                SELECT value FROM {self._fq_table}
                WHERE scope = $1 AND key = $2
                """,
                self._user_id,
                key,
            )
            if row is None:
                return None
            value = row["value"]
            return json.loads(value) if isinstance(value, str) else value

        try:
            return self._oneshot_sync(_op)
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    def write_sync(self, key: str, value: Any) -> None:
        if not isinstance(self._url, str):
            from loomable.kernel.stores import _run_sync

            _run_sync(self.write(key, value))
            return

        payload = json.dumps(value, default=str)

        async def _op(conn: Any) -> None:
            await conn.execute(
                f"""
                INSERT INTO {self._fq_table} (scope, key, value)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (scope, key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """,
                self._user_id,
                key,
                payload,
            )

        try:
            self._oneshot_sync(_op)
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc


class PgVectorBackend(_PoolMixin):
    """Vector memory backed by PostgreSQL (float arrays + cosine similarity).

    Does **not** require the ``pgvector`` extension — embeddings are stored as
    ``DOUBLE PRECISION[]`` and ranked in Python for portability. Suitable for
    moderate corpora; swap for native pgvector SQL when you need ANN at scale.
    """

    def __init__(
        self,
        url: str | Any,
        *,
        dimensions: int = 1536,
        user_id: str | None = None,
        table: str = "loomable_vectors",
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        super().__init__(url, min_size=min_size, max_size=max_size)
        self.dimensions = int(dimensions)
        self._user_id = user_id or ""
        self._table = table
        self._schema = schema
        self._backend_id = f"postgres-vector:{schema}.{table}:{self._user_id or '_'}"

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def _fq_table(self) -> str:
        return f'"{self._schema}"."{self._table}"'

    async def setup(self) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._fq_table} (
                    scope TEXT NOT NULL,
                    id TEXT NOT NULL,
                    embedding DOUBLE PRECISION[] NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scope, id)
                )
                """
            )
        self._ready = True

    async def _ensure_ready(self) -> Any:
        pool = await self._ensure_pool()
        if not self._ready:
            await self.setup()
        return pool

    def _validate_dims(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"PgVectorBackend expected {self.dimensions} dims, got {len(vector)}"
            )

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._validate_dims(vector)
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._fq_table} (scope, id, embedding, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (scope, id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    self._user_id,
                    id,
                    [float(x) for x in vector],
                    json.dumps(metadata or {}, default=str),
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        self._validate_dims(vector)
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, embedding, metadata FROM {self._fq_table}
                    WHERE scope = $1
                    """,
                    self._user_id,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in rows:
            emb = list(row["embedding"] or [])
            meta = row["metadata"] or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            score = _cosine(vector, [float(x) for x in emb])
            scored.append((score, str(row["id"]), dict(meta)))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, item_id, metadata in scored[: max(0, int(k))]:
            meta = dict(metadata)
            meta.pop("score", None)
            out.append({**meta, "id": item_id, "score": score})
        return out

    async def delete(self, id: str) -> None:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    DELETE FROM {self._fq_table}
                    WHERE scope = $1 AND id = $2
                    """,
                    self._user_id,
                    id,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc

    async def get(self, id: str) -> dict[str, Any] | None:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT id, metadata FROM {self._fq_table}
                    WHERE scope = $1 AND id = $2
                    """,
                    self._user_id,
                    id,
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc
        if row is None:
            return None
        meta = row["metadata"] or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta = dict(meta)
        meta.pop("score", None)
        return {**meta, "id": str(row["id"])}

    async def scan(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, metadata FROM {self._fq_table}
                    WHERE scope = $1
                    LIMIT $2
                    """,
                    self._user_id,
                    max(0, int(limit)),
                )
        except Exception as exc:  # noqa: BLE001
            raise MemoryBackendError(self._backend_id) from exc
        out: list[dict[str, Any]] = []
        for row in rows:
            meta = row["metadata"] or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta = dict(meta)
            meta.pop("score", None)
            out.append({**meta, "id": str(row["id"])})
        return out
