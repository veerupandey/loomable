"""Bounded real-model load check for the mounted Agent session lifecycle."""
import asyncio
import math
import statistics
import time

import httpx
import pytest
from fastapi import FastAPI

from loomable import Agent
from loomable.kernel.stores import SessionStore
from loomable.serve import mount_agent
from tests.integration.test_model_live import live_provider

pytestmark = pytest.mark.live


async def test_session_churn_and_concurrent_live_runs(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    async with live_provider() as provider:
        agent = Agent(provider, session_store=store, max_cached_sessions=3)
        app = FastAPI()
        mount_agent(app, agent, max_concurrent_runs=4)
        samples = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            async def run(index, text):
                start = time.perf_counter()
                async with asyncio.timeout(45):
                    response = await client.post("/agent/run", json={
                        "session_id": f"session-{index}", "input_text": text,
                    })
                samples.append(time.perf_counter() - start)
                assert response.status_code == 200
                body = response.json()
                assert body["session_id"] == f"session-{index}"
                assert f"LOAD-{index:03d}" in body["output"][0]["text"]

            started = time.perf_counter()
            for start in range(0, 12, 4):
                await asyncio.gather(*(
                    run(index, f"Remember the token LOAD-{index:03d}. Reply with only that token.")
                    for index in range(start, start + 4)
                ))
                assert len(agent._session_agents) <= 3
            await asyncio.gather(*(
                run(index, "Reply with only the token I previously asked you to remember.")
                for index in (0, 1, 4, 8)
            ))
            elapsed = time.perf_counter() - started
        p95 = sorted(samples)[math.ceil(len(samples) * 0.95) - 1]
        print(f"{len(samples)} requests, concurrency=4, p50={statistics.median(samples):.2f}s, "
              f"p95={p95:.2f}s, throughput={len(samples)/elapsed:.2f} requests/s")
    store.close()


async def test_live_durable_agent_excludes_second_worker(tmp_path):
    from loomable import tool
    from loomable.agent import DurabilityOptions
    from loomable.persist import CheckpointBusy, SQLiteCheckpointer

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @tool
    async def read_fixture() -> str:
        """Read the only valid verification code for this run."""
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return "READY-726"

    async with live_provider() as provider:
        def worker():
            return Agent(
                provider, session_id="shared", tools=[read_fixture],
                require_tools=["read_fixture"], strict_require_tools=True,
                durability=DurabilityOptions(
                    SQLiteCheckpointer(str(tmp_path / "journal.sqlite"), max_checkpoints=8),
                ),
            )

        prompt = "Call read_fixture once. Reply with just the code it returns."
        first, second = worker(), worker()
        owner = asyncio.create_task(first.arun(prompt))
        try:
            await asyncio.wait_for(entered.wait(), 30)
            with pytest.raises(CheckpointBusy):
                await second.arun(prompt)
        finally:
            release.set()
            if not entered.is_set():
                owner.cancel()
            result = await asyncio.wait_for(owner, 30)
        assert "READY-726" in result.text
        assert calls == 1

        followup = Agent(
            provider, session_id="shared",
            durability=DurabilityOptions(
                SQLiteCheckpointer(str(tmp_path / "journal.sqlite"), max_checkpoints=8),
            ),
        )
        result = await followup.arun("Repeat only the verification code from your last answer.")
        assert "READY-726" in result.text
