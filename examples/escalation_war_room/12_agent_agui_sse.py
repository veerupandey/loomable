"""Live gate — Agent AG-UI SSE over FastAPI (Gemini).

    python 12_agent_agui_sse.py

Asserts text/event-stream with RUN_STARTED → TEXT_* → RUN_FINISHED.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loomable import Agent
from loomable.serve import mount_agent

from _common import ESCALATION_EMAIL, OUTPUT, make_provider

ROOT = Path(__file__).resolve().parent


async def main() -> None:
    from fastapi import FastAPI
    import httpx

    agent = Agent(
        model=make_provider(),
        role="War Room Triage",
        goal="Summarize INC-88421 for the bridge",
        modalities="text",
        session_id="inc-88421-agui",
    )
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent")

    body = {
        "messages": [
            {
                "role": "user",
                "parts": [
                    {
                        "modality": "text",
                        "text": (
                            "In 3 short bullets, triage this page. "
                            "Mention INC-88421 and SEV-1.\n\n"
                            + ESCALATION_EMAIL[:1200]
                        ),
                    }
                ],
            }
        ]
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180.0) as client:
        health = await client.get("/agent/health")
        assert health.status_code == 200, health.text

        resp = await client.post("/agent/run/events", json=body)
        assert resp.status_code == 200, resp.text
        ctype = resp.headers.get("content-type", "")
        assert "text/event-stream" in ctype, ctype
        text = resp.text
        print(text[:2500])
        assert "event: RUN_STARTED" in text
        assert "event: RUN_FINISHED" in text
        assert "TEXT_MESSAGE" in text or "INC-88421" in text or "SEV" in text

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "agent_agui_sse.txt"
    out.write_text(text)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
