"""Live gate — Case + WorkItems board + AG-UI SSE (Gemini).

    python 11_case_sse.py

Uses Case(board=True, dispatch='reuse', accept=...) and streams AG-UI events
including STATE_SNAPSHOT / STATE_DELTA and NODE_*.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from loomable import Case, tool
from loomable.flow.loop import VerdictResult
from loomable.serve import mount_case
from loomable.stream import NODE_STARTED, RUN_FINISHED, RUN_STARTED, STATE_DELTA, STATE_SNAPSHOT

from _common import ESCALATION_EMAIL, OUTPUT, ROOT, make_provider

WORK = ROOT / "workspace_case"


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up AcmePay incident ticket."""
    return json.dumps(
        {
            "id": ticket_id.upper(),
            "severity": "P1",
            "service": "settlement-rail-v3",
            "customer": "BharatNova",
        }
    )


def accept_sev(output, context) -> VerdictResult:
    text = output.text() or ""
    ok = ("SEV-" in text) and ("INC-" in text.upper() or "88421" in text)
    detail = "" if ok else "Need SEV-* and INC-88421"
    return VerdictResult(ok=ok, detail=detail)


async def run_case_stream() -> list[str]:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    case = Case(
        model=make_provider(),
        goal="Close INC-88421 with a short SEV packet + customer update",
        board=True,
        dispatch="reuse",
        accept=accept_sev,
        max_rounds=3,
        max_steps=3,
        tools=[lookup_ticket],
        modalities="text",
        name="war-room-case",
        session_id="inc-88421-case",
    )

    types: list[str] = []
    texts: list[str] = []
    async for ev in case.astream_events(
        "Produce a war-room escalation answer. Must include SEV-* and INC-88421.\n\n"
        + ESCALATION_EMAIL
    ):
        types.append(ev.type)
        print(f"  event: {ev.type}")
        if ev.type == STATE_DELTA:
            print(f"    board items={len((ev.data.get('board') or {}).get('items') or [])}")
        if ev.type == RUN_FINISHED:
            texts.append(json.dumps(ev.data)[:500])

    assert RUN_STARTED in types
    assert STATE_SNAPSHOT in types
    assert STATE_DELTA in types
    assert NODE_STARTED in types
    assert RUN_FINISHED in types
    assert case.board is not None
    assert len(case.board.list()) >= 1
    print(f"[case stream] events={len(types)} board={len(case.board.list())} items")
    return types


async def run_case_fastapi_sse() -> str:
    from fastapi import FastAPI
    import httpx

    case = Case(
        model=make_provider(),
        goal="Triage INC-88421",
        board=True,
        dispatch="reuse",
        accept=accept_sev,
        max_rounds=2,
        max_steps=2,
        tools=[lookup_ticket],
        modalities="text",
        name="case-sse",
    )
    app = FastAPI()
    mount_case(app, case, prefix="/cases")

    body = {
        "messages": [
            {
                "role": "user",
                "parts": [
                    {
                        "modality": "text",
                        "text": (
                            "Short SEV packet for INC-88421 with SEV-1 label.\n\n"
                            + ESCALATION_EMAIL[:1000]
                        ),
                    }
                ],
            }
        ]
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=240.0) as client:
        health = await client.get("/cases/health")
        assert health.status_code == 200
        resp = await client.post("/cases/run/events", json=body)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        text = resp.text
        assert "event: RUN_STARTED" in text
        assert "event: RUN_FINISHED" in text
        print(text[:2000])
        return text


async def main() -> None:
    types = await run_case_stream()
    sse = await run_case_fastapi_sse()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "case_sse.txt"
    path.write_text(
        "types=" + ",".join(types) + "\n\n--- /cases/run/events ---\n\n" + sse
    )
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
