"""Verify completed-session retention against PostgreSQL without model calls."""
import os
from uuid import uuid4

import pytest

from loomable.persist import Checkpoint, PostgresCheckpointer


async def test_retention_prunes_whole_threads_and_preserves_unfinished_runs():
    dsn = os.getenv("LOOMABLE_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("configure a disposable PostgreSQL database")
    asyncpg = pytest.importorskip("asyncpg")

    table = f"retention_{uuid4().hex}"
    store = PostgresCheckpointer(dsn, table=table, max_size=1)
    worker = PostgresCheckpointer(dsn, table=table, max_size=1)
    db = await asyncpg.connect(dsn)
    try:
        await store.setup()
        expired = "agent-session-expired"
        claimed = "agent-session-owned"
        restarted = "agent-session-restarted"
        for thread in (expired, claimed, restarted, "workflow-complete"):
            await store.put(Checkpoint(thread_id=thread, step=0))
            await store.put(Checkpoint(thread_id=thread, step=1, complete=True))
        await db.execute(f'UPDATE "{table}" SET timestamp = timestamp - 3600')
        await store.put(Checkpoint(thread_id=restarted, step=0, complete=False))
        await store.put(Checkpoint(thread_id="agent-session-recent", step=1, complete=True))

        async with worker.claim(claimed):
            assert await store.prune_completed_agent_sessions(600) == 1
            assert await store.list(expired) == []
            assert len(await store.list(claimed)) == 2
        assert await store.prune_completed_agent_sessions(600, limit=1) == 1
        assert await store.list(claimed) == []
        assert not (await store.get(restarted)).complete
        assert (await store.get("workflow-complete")).complete
        assert (await store.get("agent-session-recent")).complete
        assert await store.prune_completed_agent_sessions(600) == 0
    finally:
        await store.aclose()
        await worker.aclose()
        try:
            await db.execute(f'DROP TABLE IF EXISTS "{table}"')
        finally:
            await db.close()
