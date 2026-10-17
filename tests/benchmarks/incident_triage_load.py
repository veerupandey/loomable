"""Measure end-to-end triage latency and PostgreSQL growth under sustained load."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from uuid import uuid4

import asyncpg
import httpx

from examples.incident_triage.__main__ import FIXTURES, Handoff, validate_handoff

EXPECTED = {
    "payments": ("SEV2", "Payments On-Call", "2%"),
    "search": ("SEV3", "Search On-Call", "250"),
}


async def storage(db):
    row = await db.fetchrow("""
        SELECT count(*) AS rows, count(DISTINCT thread_id) AS sessions,
               count(*) FILTER (WHERE complete) AS completed,
               pg_total_relation_size('triage_checkpoints') AS bytes
        FROM triage_checkpoints
    """)
    return dict(row)


async def benchmark(url, duration, rate):
    fixtures = [(json.loads((FIXTURES / f"{name}.json").read_text()), expected)
                for name, expected in EXPECTED.items()]
    statuses = Counter()
    latencies = []
    pending = set()
    db = await asyncpg.connect(os.environ["TRIAGE_DATABASE_URL"], command_timeout=10)
    headers = {"Authorization": f"Bearer {os.environ['TRIAGE_ACCESS_TOKEN']}"}
    try:
        initial = await storage(db)
        print(json.dumps({"start": initial}), flush=True)
        async with httpx.AsyncClient(base_url=url, headers=headers, timeout=32) as client:
            workers = await asyncio.gather(*(
                client.get("/health", headers={"Connection": "close"}) for _ in range(12)
            ))
            worker_ids = {r.raise_for_status().json()["worker"] for r in workers}
            if len(worker_ids) != 2:
                raise RuntimeError(f"Expected two workers, found {len(worker_ids)}")

            async def submit(i):
                packet, (severity, owner, threshold) = fixtures[i % len(fixtures)]
                start = time.monotonic()
                try:
                    response = await client.post("/triage", json=packet | {"request_id": str(uuid4())})
                    elapsed = time.monotonic() - start
                    if response.status_code == 200:
                        handoff = validate_handoff(Handoff.model_validate_json(response.content), packet)
                        if ((handoff.severity, handoff.owner) != (severity, owner)
                                or threshold not in handoff.verification or "M1" not in handoff.evidence_ids):
                            raise ValueError("Fixture handoff does not match the runbook")
                        latencies.append(elapsed)
                    statuses[str(response.status_code)] += 1
                except httpx.HTTPError as exc:
                    statuses[type(exc).__name__] += 1
                except ValueError:
                    statuses["invalid_handoff"] += 1

            start = time.monotonic()
            report_at = start + 60
            for i in range(math.ceil(rate * duration)):
                await asyncio.sleep(max(0, start + i / rate - time.monotonic()))
                if len(pending) < 32:
                    task = asyncio.create_task(submit(i))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                else:
                    statuses["client_capacity"] += 1
                if time.monotonic() >= report_at:
                    print(json.dumps({"elapsed": round(time.monotonic() - start),
                                      "statuses": statuses, "storage": await storage(db)}), flush=True)
                    report_at += 60
            await asyncio.gather(*pending)
            final = await storage(db)
        ordered = sorted(latencies)
        percentiles = {f"p{p}": round(ordered[math.ceil(p * len(ordered) / 100) - 1], 3)
                       for p in (50, 95, 99)} if ordered else {}
        print(json.dumps({
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "workers": sorted(worker_ids),
            "offered_requests_per_second": rate,
            "statuses": statuses,
            "successful_latency_seconds": percentiles,
            "start": initial,
            "end": final,
            "growth": {k: final[k] - initial[k] for k in initial},
        }), flush=True)
        return not latencies or any(status not in {"200", "429"} for status in statuses)
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8874")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--rate", type=float, default=0.75)
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("duration and rate must be positive and finite")
    raise SystemExit(asyncio.run(benchmark(args.url, args.seconds, args.rate)))


if __name__ == "__main__":
    main()
