"""Exercise HTTP admission with a deterministic workflow and no model."""
import asyncio

import httpx
from fastapi import FastAPI

from loomable import Workflow
from loomable.serve import mount_agent


async def test_sse_run_holds_capacity_until_stream_finishes():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def step(value):
        entered.set()
        await release.wait()
        return value

    app = FastAPI()
    mount_agent(app, Workflow("admission").step("work", step),
                max_concurrent_runs=1, api_key="local-test")
    headers = {"Authorization": "Bearer local-test"}
    payload = {"input_text": "ready"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/agent/run/events", json=payload, headers=headers))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            denied = await client.post("/agent/run", json=payload)
            assert denied.status_code == 401
            busy = await client.post("/agent/run", json=payload, headers=headers)
            assert busy.status_code == 429
            assert busy.headers["Retry-After"] == "1"
        finally:
            release.set()
            streamed = await asyncio.wait_for(first, 2)
        assert "event: RUN_FINISHED" in streamed.text
        response = await client.post("/agent/run", json=payload, headers=headers)
        assert response.status_code == 200
