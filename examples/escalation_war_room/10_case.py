"""Case gate — plan → dispatch → synthesize → accept.

    from loomable import Case, Agent

    case = Case(model=..., dispatch="spawn", accept=..., max_rounds=3)
    result = await case.arun("Handle INC-88421")

    agent = Agent(model=..., mode="case", dispatch="reuse", accept=...)
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from loomable import Agent, Case, tool
from loomable.flow.loop import VerdictResult

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
            "linked_change": "CHG-55219",
        }
    )


def accept_sev_packet(output, context) -> VerdictResult:
    text = output.text() or ""
    ok = ("SEV-" in text) and ("INC-" in text.upper() or "88421" in text)
    detail = ""
    if "SEV-" not in text:
        detail = "Must include a SEV-* severity label"
    elif "INC-" not in text.upper() and "88421" not in text:
        detail = "Must reference the incident id"
    return VerdictResult(ok=ok, detail=detail)


async def run_case_spawn() -> str:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    case = Case(
        model=make_provider(),
        goal="Produce a short war-room escalation answer",
        board=True,
        dispatch="spawn",
        accept=accept_sev_packet,
        max_rounds=3,
        max_steps=3,
        tools=[lookup_ticket],
        modalities="text",
        name="war-room-case",
        session_id="inc-88421-case",
    )
    result = await case.arun(
        "Must include SEV-* and INC-88421.\n\n" + ESCALATION_EMAIL
    )
    print(
        f"[case spawn] verified={result.metadata.get('loop_verified')} "
        f"iters={result.metadata.get('loop_iterations')} "
        f"dispatch={result.metadata.get('dispatch')} "
        f"board={len((result.metadata.get('board') or {}).get('items') or [])}"
    )
    print(result.output.text()[:2000])
    assert result.metadata.get("case") is True
    assert "SEV-" in result.output.text()
    return result.output.text()


async def run_agent_mode_case() -> str:
    agent = Agent(
        model=make_provider(),
        mode="case",
        dispatch="reuse",
        accept=accept_sev_packet,
        max_rounds=3,
        max_plan_steps=3,
        tools=[lookup_ticket],
        modalities="text",
        session_id="inc-88421-agent-case",
    )
    result = await agent.arun(
        "Escalate INC-88421 for BharatNova. Include SEV-* and incident id.\n\n"
        + ESCALATION_EMAIL
    )
    print(
        f"[agent mode=case] verified={result.metadata.get('loop_verified')} "
        f"chars={len(result.output.text() or '')}"
    )
    print(result.output.text()[:1600])
    assert result.metadata.get("case") is True
    assert "SEV-" in result.output.text()
    return result.output.text()


async def main() -> None:
    spawn_text = await run_case_spawn()
    agent_text = await run_agent_mode_case()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "case_packet.md"
    path.write_text(
        "# Case outputs\n\n## dispatch=spawn\n\n"
        + spawn_text
        + "\n\n## Agent(mode='case')\n\n"
        + agent_text
        + "\n"
    )
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
