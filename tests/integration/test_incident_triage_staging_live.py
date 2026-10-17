"""Exercise staging fault handling while forwarding healthy calls to real OpenAI."""
import asyncio
import hashlib
import json
import os
import secrets
import socket
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response

from examples.incident_triage.__main__ import build_triage_agent
from loomable.agent import DurabilityOptions
from loomable.agent.durable import session_key
from loomable.persist import PostgresCheckpointer

pytestmark = pytest.mark.live
FIXTURE = Path("examples/incident_triage/fixtures/payments.json")


def loopback_socket():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    return listener


async def wait_ready(client, url):
    async with asyncio.timeout(15):
        while True:
            try:
                response = await client.get(f"{url}/health")
                if response.is_success:
                    return response.json()["worker"]
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.05)


async def stop_worker(worker):
    if worker.returncode is None:
        worker.terminate()
    try:
        await asyncio.wait_for(worker.wait(), 5)
    except TimeoutError:
        worker.kill()
        await worker.wait()


async def test_two_worker_triage_recovers_and_sheds_load():
    dsn = os.getenv("LOOMABLE_TEST_POSTGRES_URL")
    if not dsn or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("requires disposable PostgreSQL and real OpenAI credentials")

    packet = json.loads(FIXTURE.read_text())
    token = secrets.token_urlsafe(48)
    submissions = [packet | {"request_id": str(uuid4())} for _ in range(5)]
    held = asyncio.Event()
    admitted = asyncio.Event()
    outage_started = asyncio.Event()
    release_outage = asyncio.Event()
    held_count = 0
    throttled = 0
    upstream_calls = []
    outages = []
    mode = "outage"

    async with AsyncExitStack() as stack:
        upstream = await stack.enter_async_context(httpx.AsyncClient(timeout=20))
        relay = FastAPI()

        @relay.post("/v1/chat/completions")
        async def forward(request: Request):
            nonlocal held_count, throttled
            payload = await request.json()
            has_tools = any(message["role"] == "tool" for message in payload["messages"])
            if mode == "outage" and has_tools:
                outage_started.set()
                await release_outage.wait()
                outages.append(True)
                return Response(status_code=503)
            if mode == "restore" and not throttled:
                throttled += 1
                return Response(status_code=429, headers={"Retry-After": "1"})
            if mode == "hold" and not has_tools:
                held_count += 1
                if held_count == 4:
                    admitted.set()
                await held.wait()
            upstream_calls.append(has_tools)
            response = await upstream.post(
                "https://api.openai.com/v1/chat/completions", json=payload,
                headers={"Authorization": request.headers["Authorization"]},
            )
            return Response(response.content, status_code=response.status_code,
                            media_type="application/json")

        listener = stack.enter_context(loopback_socket())
        relay_port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(relay, log_level="warning", access_log=False))
        relay_task = asyncio.create_task(server.serve(sockets=[listener]))

        async def stop_relay():
            held.set()
            release_outage.set()
            server.should_exit = True
            await relay_task

        stack.push_async_callback(stop_relay)
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)

        client = await stack.enter_async_context(httpx.AsyncClient(
            timeout=35, headers={"Authorization": f"Bearer {token}"},
        ))
        urls = []
        pids = set()
        for _ in range(2):
            with loopback_socket() as worker_socket:
                port = worker_socket.getsockname()[1]
                worker = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "uvicorn", "examples.incident_triage.staging:app",
                    "--fd", str(worker_socket.fileno()), "--log-level", "warning",
                    pass_fds=(worker_socket.fileno(),),
                    env=os.environ | {
                        "TRIAGE_DATABASE_URL": dsn,
                        "TRIAGE_ACCESS_TOKEN": token,
                        "TRIAGE_OPENAI_BASE_URL": f"http://127.0.0.1:{relay_port}/v1",
                        "TRIAGE_RETENTION_INTERVAL_SECONDS": "1",
                    },
                )
            stack.push_async_callback(stop_worker, worker)
            urls.append(f"http://127.0.0.1:{port}")
            pids.add(await wait_ready(client, urls[-1]))
        assert len(pids) == 2

        async def submit(worker, submission):
            return await client.post(f"{urls[worker]}/triage", json=submission)

        submission = submissions[0]
        unauthorized = await client.post(f"{urls[0]}/triage", json=submission,
                                         headers={"Authorization": "Bearer invalid"})
        assert unauthorized.status_code == 401
        oversized = await client.post(f"{urls[0]}/triage", content=b" " * (128 * 1024 + 1))
        assert oversized.status_code == 413
        digest = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
        session_id = f"triage:{submission['request_id']}:{digest}"
        thread_id = session_key(session_id)
        store = PostgresCheckpointer(dsn, table="triage_checkpoints")
        stack.push_async_callback(store.aclose)
        interrupted = asyncio.create_task(submit(0, submission))
        try:
            await asyncio.wait_for(outage_started.wait(), 10)
            conflict = await submit(1, submission)
            assert conflict.status_code == 409
        finally:
            release_outage.set()
            failed = await interrupted
        assert failed.status_code == 503 and failed.headers["Retry-After"] == "1"
        assert len(outages) == 3
        before = await store.get(thread_id)
        assert not before.complete
        assert all(batch["status"] == "done" for batch in before.session_state["batches"])
        assert before.session_state["batches"]

        mode = "restore"
        first_restored = len(upstream_calls)
        resumed = await submit(1, submission)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["severity"] == "SEV2"
        assert resumed.json()["owner"] == "Payments On-Call"
        assert upstream_calls[first_restored] and throttled == 1
        after = await store.get(thread_id)
        assert after.complete and after.session_state["run_id"] == before.session_state["run_id"]
        assert after.session_state["batches"][:len(before.session_state["batches"])] == before.session_state["batches"]
        upstream_count = len(upstream_calls)
        duplicate = await submit(0, submission)
        assert duplicate.status_code == 200 and duplicate.json() == resumed.json()
        assert len(upstream_calls) == upstream_count
        independent_agent = build_triage_agent(
            "openai:gpt-5.4-mini", packet, session_id=session_id, durability=DurabilityOptions(store),
        )
        with pytest.raises(ValueError, match="different input"):
            await independent_agent.arun_once("A different request")
        with pytest.raises(ValueError, match="new session_id"):
            await independent_agent.arun(f"Triage incident {packet['incident']}.")
        replayed = await independent_agent.arun_once(f"Triage incident {packet['incident']}.")
        assert replayed.structured.model_dump() == resumed.json()
        assert replayed.metadata == {"run_id": after.session_state["run_id"], "replayed": True}
        assert replayed.usage == after.session_state["response"]["usage"]

        mode = "hold"
        accepted = [asyncio.create_task(submit(i % 2, data)) for i, data in enumerate(submissions[1:])]
        try:
            await asyncio.wait_for(admitted.wait(), 5)
            rejected = await asyncio.gather(*(submit(i % 2, submission) for i in range(8)))
            assert all(r.status_code == 429 and r.headers["Retry-After"] == "1" for r in rejected)
            assert set(await asyncio.gather(*(wait_ready(client, url) for url in urls))) == pids
        finally:
            held.set()
            results = await asyncio.gather(*accepted)
        assert all(r.status_code == 200 for r in results), [r.text for r in results]
        assert (await submit(1, submission)).status_code == 200

        import asyncpg

        db = await asyncpg.connect(dsn)
        stack.push_async_callback(db.close)
        await db.execute("UPDATE triage_checkpoints SET timestamp = 0 WHERE thread_id = $1", thread_id)
        async with asyncio.timeout(5):
            while await store.get(thread_id) is not None:
                await asyncio.sleep(0.05)
