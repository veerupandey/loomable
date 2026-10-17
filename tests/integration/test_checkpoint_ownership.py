"""Exercise checkpoint ownership without a model."""
import asyncio
import os
import sys

import pytest

from loomable.persist import Checkpoint, CheckpointBusy, SQLiteCheckpointer


async def test_sqlite_owner_is_exclusive_and_released_on_worker_exit(tmp_path):
    path = str(tmp_path / "checkpoint.db")
    checkpointer = SQLiteCheckpointer(path)
    script = """
import asyncio, sys
from loomable.persist import SQLiteCheckpointer
async def main():
    async with SQLiteCheckpointer(sys.argv[1]).claim('thread'):
        print('owned', flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
"""
    worker = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, path, stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(worker.stdout.readline(), 5) == b"owned\n"
        with pytest.raises(CheckpointBusy):
            async with checkpointer.claim("thread"):
                pytest.fail("a second worker acquired the thread")
        async with checkpointer.claim("other-thread"):
            await checkpointer.put(Checkpoint(thread_id="other-thread", step=1))
    finally:
        worker.kill()
        await worker.wait()
    async with checkpointer.claim("thread"):
        await checkpointer.put(Checkpoint(thread_id="thread", step=1))


@pytest.mark.live
async def test_postgres_owners_share_neither_lock_nor_pool_connection():
    from loomable.persist import PostgresCheckpointer

    dsn = os.getenv("LOOMABLE_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("set LOOMABLE_TEST_POSTGRES_URL")
    first = PostgresCheckpointer(dsn, max_size=1)
    second = PostgresCheckpointer(dsn, max_size=1)
    try:
        await asyncio.gather(first.setup(), second.setup())
        async with asyncio.timeout(10):
            async with first.claim("test-thread"):
                await first.put(Checkpoint(thread_id="test-thread", step=1))
                with pytest.raises(CheckpointBusy):
                    async with second.claim("test-thread"):
                        pytest.fail("a second connection acquired the thread")
            async with second.claim("test-thread"):
                assert (await second.get("test-thread")).step == 1
    finally:
        await first.close()
        await second.close()
