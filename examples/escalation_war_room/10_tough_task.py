"""Tough-task gate — plan → fan-out spawn → synthesize → verify.

Shows the easy mode for hard agent work without low-level Flow graphs::

    from loomable import ToughTask, Agent

    task = ToughTask(model=..., fan_out="spawn", verify=..., max_iterations=3)
    result = await task.arun("Handle INC-88421")

    # Or force an Agent into the same pipeline:
    agent = Agent(model=..., mode="tough", fan_out="spawn", verifier=...)
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from loomable import Agent, ToughTask, tool
from loomable.flow.loop import VerdictResult

from _common import ESCALATION_EMAIL, OUTPUT, ROOT, make_provider

WORK = ROOT / "workspace_tough"


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up AcmePay incident ticket."""
    return json.dumps(
        {
            "id": ticket_id.upper(),
            "severity": "P1",
            "service": "settlement-rail-v3",
            "customer": "BharatNova",
            "linked_change": "CHG-55219",
        }
    )


def verify_sev_packet(output, context) -> VerdictResult:
    text = output.text() or ""
    ok = ("SEV-" in text) and ("INC-" in text.upper() or "88421" in text)
    detail = ""
    if "SEV-" not in text:
        detail = "Must include a SEV-* severity label"
    elif "INC-" not in text.upper() and "88421" not in text:
        detail = "Must reference the incident id"
    return VerdictResult(ok=ok, detail=detail)


async def run_tough_spawn() -> str:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    task = ToughTask(
        model=make_provider(),
        fan_out="spawn",
        verify=verify_sev_packet,
        max_iterations=3,
        max_steps=3,
        tools=[lookup_ticket],
        modalities="text",
        name="war-room-tough",
        session_id="inc-88421-tough",
    )
    result = await task.arun(
        "Produce a short war-room escalation answer for this SEV page.\n"
        "Must include SEV-* and INC-88421.\n\n"
        + ESCALATION_EMAIL
    )
    print(
        f"[tough spawn] verified={result.metadata.get('loop_verified')} "
        f"iters={result.metadata.get('loop_iterations')} "
        f"fan_out={result.metadata.get('fan_out')}"
    )
    print(result.output.text()[:2000])
    assert result.metadata.get("tough") is True
    assert "SEV-" in result.output.text()
    return result.output.text()


async def run_agent_mode_tough() -> str:
    agent = Agent(
        model=make_provider(),
        mode="tough",
        fan_out="map",
        verifier=verify_sev_packet,
        max_verify_retries=2,
        max_plan_steps=3,
        tools=[lookup_ticket],
        modalities="text",
        session_id="inc-88421-agent-tough",
    )
    result = await agent.arun(
        "Escalate INC-88421 for BharatNova. Include SEV-* and incident id.\n\n"
        + ESCALATION_EMAIL
    )
    print(
        f"[agent mode=tough] verified={result.metadata.get('loop_verified')} "
        f"chars={len(result.output.text() or '')}"
    )
    print(result.output.text()[:1600])
    assert result.metadata.get("tough") is True
    assert "SEV-" in result.output.text()
    return result.output.text()


async def main() -> None:
    spawn_text = await run_tough_spawn()
    agent_text = await run_agent_mode_tough()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "tough_task_packet.md"
    path.write_text(
        "# Tough task outputs\n\n## spawn fan-out\n\n"
        + spawn_text
        + "\n\n## Agent(mode='tough')\n\n"
        + agent_text
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {path}")
    print("[ok] TOUGH TASK GATE PASSED")


if __name__ == "__main__":
    asyncio.run(main())
