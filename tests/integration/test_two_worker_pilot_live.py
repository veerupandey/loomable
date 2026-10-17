"""Paced live-model pilot with real HTTP workers and shared PostgreSQL."""
import asyncio
import math
import os
import statistics
import sys
import time
from contextlib import suppress

import httpx
import pytest

from loomable.agent.durable import session_key
from loomable.persist import PostgresCheckpointer

pytestmark = pytest.mark.live


async def wait_for_path(path):
    async with asyncio.timeout(40):
        while not path.exists() or path.stat().st_size == 0:
            await asyncio.sleep(0.05)
    return path.read_text().strip()


async def rss_mib(pid):
    proc = await asyncio.create_subprocess_exec(
        "ps", "-o", "rss=", "-p", str(pid), stdout=asyncio.subprocess.PIPE,
    )
    data, _ = await proc.communicate()
    return int(data.strip()) / 1024


async def test_two_worker_postgres_pilot(tmp_path):
    dsn = os.environ.get("LOOMABLE_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("configure a disposable PostgreSQL database")
    duration = float(os.getenv("LOOMABLE_PILOT_SECONDS", "1800"))
    interval = float(os.getenv("LOOMABLE_PILOT_INTERVAL", "30"))
    assert 0 < duration <= 1800 and interval >= 2
    workers = []
    retired_rss = []
    store = PostgresCheckpointer(dsn)
    samples, rss, errors, pool_samples = [], [], [], []
    try:
        async def start_worker(index):
            port_path = tmp_path / f"worker-{index}.port"
            port_path.unlink(missing_ok=True)
            worker = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "tests.integration.pilot_worker",
                str(tmp_path), str(port_path), stderr=asyncio.subprocess.DEVNULL,
            )
            return worker, port_path

        urls = []
        for index in range(2):
            worker, port_path = await start_worker(index)
            workers.append(worker)
            urls.append(f"http://127.0.0.1:{await wait_for_path(port_path)}")

        async with httpx.AsyncClient(timeout=50) as client:
            async def run(url, session, prompt, *, route="/agent"):
                start = time.monotonic()
                response = await client.post(url + route + "/run", json={
                    "session_id": session, "input_text": prompt,
                })
                return response, time.monotonic() - start

            async def sample(worker, index):
                token = f"PILOT-{index:04d}"
                try:
                    first, latency = await run(urls[worker], token, f"The public inventory label for this test is {token}. Reply with just the label.")
                    assert first.status_code == 200, (first.status_code, first.text[:500])
                    samples.append(latency)
                    assert token in first.json()["output"][0]["text"]
                    second, latency = await run(urls[1-worker], token, "What public inventory label did I give you in the previous message? Reply with just the label.")
                    assert second.status_code == 200, (second.status_code, second.text[:500])
                    samples.append(latency)
                    assert token in second.json()["output"][0]["text"]
                except Exception as exc:
                    detail = str(exc) if isinstance(exc, AssertionError) else type(exc).__name__
                    errors.append(f"{token}: {detail}")
                    print(f"Pilot request failed for {token}: {detail}", flush=True)

            async def traffic_during_recovery():
                response, latency = await run(urls[1], "overlap", "Reply with just PILOT-OVERLAP.")
                assert response.status_code == 200
                assert "PILOT-OVERLAP" in response.json()["output"][0]["text"]
                samples.append(latency)

            async def crash_and_recover():
                started = asyncio.create_task(run(
                    urls[0], "crash", "Call record_receipt for incident INC-PILOT once, then reply with its receipt code.",
                    route="/receipt",
                ))
                try:
                    await wait_for_path(tmp_path / "receipts")
                    cp = await store.get(session_key("crash"))
                    assert cp.session_state["batches"][-1]["status"] == "running"
                    run_id = cp.session_state["run_id"]
                    competitor, _ = await run(urls[1], "crash", "Continue the incident.", route="/receipt")
                    assert competitor.status_code == 409
                    retired_rss.append(await rss_mib(workers[0].pid))
                    workers[0].kill()
                    await workers[0].wait()
                    blocked = await client.post(urls[1] + "/pilot/recover", params={"run_id": run_id})
                    assert blocked.status_code == 409
                    call_id, = blocked.json()["call_ids"]
                    resumed = await client.post(urls[1] + "/pilot/recover",
                                                params={"run_id": run_id, "call_id": call_id})
                    assert resumed.status_code == 200
                    assert "RECEIPT-726" in resumed.json()["text"]
                    assert (tmp_path / "receipts").read_text().splitlines() == ["RECEIPT-726"]
                finally:
                    started.cancel()
                    with suppress(asyncio.CancelledError, httpx.HTTPError):
                        await started

            start = time.monotonic()
            batches = 0
            next_progress = 60
            recovered = False
            while True:
                rss.append([await rss_mib(w.pid) for w in workers])
                requests = [asyncio.create_task(sample(i, batches * 2 + i)) for i in range(2)]
                await asyncio.sleep(0.1)
                pool_samples.extend([(await client.get(url + "/pilot/metrics")).json() for url in urls])
                await asyncio.gather(*requests)
                batches += 1
                elapsed = time.monotonic() - start
                if not recovered and elapsed >= duration / 2:
                    await asyncio.gather(crash_and_recover(), traffic_during_recovery())
                    worker, port_path = await start_worker(0)
                    workers[0] = worker
                    urls[0] = f"http://127.0.0.1:{await wait_for_path(port_path)}"
                    recovered = True
                    print("Worker killed during tool execution, reconciled, and replaced", flush=True)
                elapsed = time.monotonic() - start
                if elapsed >= next_progress:
                    print(f"Pilot progress {elapsed:.0f}s, {len(samples)} successful responses, "
                          f"{len(errors)} errors, RSS MiB {rss[-1]}", flush=True)
                    next_progress += 60
                remaining = duration - elapsed
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, interval))
                if time.monotonic() - start >= duration:
                    break
            rss.append([await rss_mib(w.pid) for w in workers])
            elapsed = time.monotonic() - start
            if not recovered:
                await crash_and_recover()

        pool = await store._ensure_ready()
        async with pool.acquire() as conn:
            rows = await conn.fetchval("SELECT count(*) FROM loomable_checkpoints")
            size = await conn.fetchval("SELECT pg_total_relation_size('loomable_checkpoints')")
        assert samples and not errors, errors
        p95 = sorted(samples)[math.ceil(len(samples) * 0.95) - 1]
        print(f"elapsed={elapsed:.1f}s requests={len(samples)} p50={statistics.median(samples):.2f}s "
              f"p95={p95:.2f}s rss_start={rss[0]} rss_end={rss[-1]} "
              f"rss_peak={[max([*x, *retired_rss] if i == 0 else x) for i, x in enumerate(zip(*rss))]} pool_peak={max(x['pool_busy'] for x in pool_samples)} "
              f"checkpoint_rows={rows} checkpoint_bytes={size} recovery=passed")
    finally:
        for worker in workers:
            if worker.returncode is None:
                worker.terminate()
                await asyncio.wait_for(worker.wait(), 10)
        await store.close()
