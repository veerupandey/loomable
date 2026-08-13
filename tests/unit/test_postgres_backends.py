"""Unit tests for Postgres backends using an in-memory fake asyncpg pool."""

from __future__ import annotations

import json
from typing import Any

import pytest

from loomable.persist.checkpoint import Checkpoint
from loomable.persist.postgres import PostgresCheckpointer
from loomable.providers.backends.postgres import PgVectorBackend, PostgresMemoryBackend


class _FakeConn:
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store

    async def execute(self, query: str, *args: Any) -> str:
        q = " ".join(query.split()).lower()
        if "create table" in q or "create index" in q:
            return "OK"
        if "insert into" in q and "loomable_checkpoints" in q:
            thread_id, step, payload, ts, complete = args
            rows = self.store.setdefault("checkpoints", [])
            rows.append(
                {
                    "id": len(rows) + 1,
                    "thread_id": thread_id,
                    "step": step,
                    "payload": json.loads(payload) if isinstance(payload, str) else payload,
                    "timestamp": ts,
                    "complete": complete,
                }
            )
            return "INSERT 0 1"
        if "delete from" in q and "loomable_checkpoints" in q and "offset" in q:
            thread_id, keep = args
            rows = [r for r in self.store.get("checkpoints", []) if r["thread_id"] == thread_id]
            rows_sorted = sorted(rows, key=lambda r: (r["timestamp"], r["id"]), reverse=True)
            keep_ids = {r["id"] for r in rows_sorted[: int(keep)]}
            self.store["checkpoints"] = [
                r
                for r in self.store.get("checkpoints", [])
                if r["thread_id"] != thread_id or r["id"] in keep_ids
            ]
            return "DELETE"
        if "insert into" in q and "loomable_kv" in q:
            scope, key, payload = args
            kv = self.store.setdefault("kv", {})
            kv[(scope, key)] = json.loads(payload) if isinstance(payload, str) else payload
            return "INSERT 0 1"
        if "delete from" in q and "loomable_kv" in q:
            scope, key = args
            self.store.setdefault("kv", {}).pop((scope, key), None)
            return "DELETE"
        if "insert into" in q and "loomable_vectors" in q:
            scope, item_id, embedding, metadata = args
            vecs = self.store.setdefault("vectors", {})
            meta = json.loads(metadata) if isinstance(metadata, str) else metadata
            vecs[(scope, item_id)] = {"embedding": list(embedding), "metadata": meta}
            return "INSERT 0 1"
        if "delete from" in q and "loomable_vectors" in q:
            scope, item_id = args
            self.store.setdefault("vectors", {}).pop((scope, item_id), None)
            return "DELETE"
        return "OK"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        q = " ".join(query.split()).lower()
        if "from" in q and "loomable_checkpoints" in q:
            thread_id = args[0]
            rows = [r for r in self.store.get("checkpoints", []) if r["thread_id"] == thread_id]
            if not rows:
                return None
            latest = sorted(rows, key=lambda r: (r["timestamp"], r["id"]), reverse=True)[0]
            return {"payload": latest["payload"]}
        if "select value from" in q and "loomable_kv" in q:
            scope, key = args
            if (scope, key) not in self.store.get("kv", {}):
                return None
            return {"value": self.store["kv"][(scope, key)]}
        if "select 1 from" in q and "loomable_kv" in q:
            scope, key = args
            if (scope, key) in self.store.get("kv", {}):
                return {"?column?": 1}
            return None
        return None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        q = " ".join(query.split()).lower()
        if "loomable_checkpoints" in q:
            thread_id = args[0]
            rows = [r for r in self.store.get("checkpoints", []) if r["thread_id"] == thread_id]
            rows = sorted(rows, key=lambda r: (r["timestamp"], r["id"]))
            return [{"payload": r["payload"]} for r in rows]
        if "loomable_vectors" in q:
            scope = args[0]
            out = []
            for (s, item_id), data in self.store.get("vectors", {}).items():
                if s == scope:
                    out.append(
                        {
                            "id": item_id,
                            "embedding": data["embedding"],
                            "metadata": data["metadata"],
                        }
                    )
            return out
        return []


class _AcquireCM:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.conn = _FakeConn(self.store)

    def acquire(self) -> _AcquireCM:
        return _AcquireCM(self.conn)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_postgres_checkpointer_put_get_list_prune() -> None:
    pool = _FakePool()
    cp = PostgresCheckpointer(pool, max_checkpoints=2)
    await cp.setup()
    await cp.put(
        Checkpoint(thread_id="t1", step=1, session_state={"a": 1}, complete=False)
    )
    await cp.put(
        Checkpoint(thread_id="t1", step=2, session_state={"a": 2}, complete=False)
    )
    await cp.put(
        Checkpoint(thread_id="t1", step=3, session_state={"a": 3}, complete=True)
    )
    latest = await cp.get("t1")
    assert latest is not None
    assert latest.session_state["a"] == 3
    assert latest.complete is True
    listed = await cp.list("t1")
    assert len(listed) == 2  # pruned to max_checkpoints
    assert listed[-1].session_state["a"] == 3


@pytest.mark.asyncio
async def test_postgres_memory_backend_scoped_kv() -> None:
    pool = _FakePool()
    alice = PostgresMemoryBackend(pool, user_id="alice")
    bob = PostgresMemoryBackend(pool, user_id="bob")
    await alice.setup()
    await bob.setup()
    await alice.write("pref", {"theme": "dark"})
    await bob.write("pref", {"theme": "light"})
    assert await alice.read("pref") == {"theme": "dark"}
    assert await bob.read("pref") == {"theme": "light"}
    assert await alice.exists("pref") is True
    await alice.delete("pref")
    assert await alice.exists("pref") is False
    assert await bob.read("pref") == {"theme": "light"}


@pytest.mark.asyncio
async def test_pgvector_backend_cosine_query() -> None:
    pool = _FakePool()
    backend = PgVectorBackend(pool, dimensions=3, user_id="u1")
    await backend.setup()
    await backend.index("a", [1.0, 0.0, 0.0], {"text": "axis-x"})
    await backend.index("b", [0.0, 1.0, 0.0], {"text": "axis-y"})
    hits = await backend.query([0.9, 0.1, 0.0], k=1)
    assert hits[0]["id"] == "a"
    assert hits[0]["text"] == "axis-x"
    await backend.delete("a")
    hits2 = await backend.query([1.0, 0.0, 0.0], k=2)
    assert all(h["id"] != "a" for h in hits2)
