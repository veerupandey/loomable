"""CHALLENGE 09 — HITL confirm + soft Team + require_tools under stress.

Agents only — no parse helpers between Team / Workflow steps.

  1) Soft Team ``coordinate`` (LLM must actually delegate)
  2) Workflow.step(..., confirm=True) HITL pause before scribe
  3) approve() + resume
  4) Scribe with require_tools write_file/write_json
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from loomable import Agent, FlowPaused, JsonFileCheckpointer, Team, Workflow, tool
from loomable.toolkits import FileTools

from _common import ESCALATION_EMAIL, OUTPUT, ROOT, make_provider

WORK = ROOT / "workspace_challenge"
CKPT = ROOT / ".checkpoints_challenge"
SESSION = "inc-88421-challenge"
OBS: list[str] = []


def note(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    OBS.append(line)


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up ticket."""
    return json.dumps(
        {
            "id": ticket_id.upper(),
            "severity": "P1",
            "service": "settlement-rail-v3",
            "linked_change": "CHG-55219",
        }
    )


class FinalPacket(BaseModel):
    incident_id: str
    customer: str
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    root_hypothesis: str
    next_actions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    approved: bool = True


def _prepare() -> Path:
    if WORK.exists():
        shutil.rmtree(WORK)
    if CKPT.exists():
        shutil.rmtree(CKPT)
    (WORK / "output").mkdir(parents=True)
    CKPT.mkdir(parents=True)
    return WORK


async def run_challenge() -> FinalPacket:
    work = _prepare()
    cp = JsonFileCheckpointer(str(CKPT))

    triage = Agent(
        model=make_provider(),
        role="Triage",
        goal="SEV + hypothesis",
        instructions="Be terse. Output severity + one root hypothesis.",
        modalities="text",
        tools=[lookup_ticket],
    )
    sla = Agent(
        model=make_provider(),
        role="SLA",
        goal="SLA risk",
        instructions="Be terse. State bridge urgency for Strategic tier.",
        modalities="text",
    )
    soft_team = Team(
        members=[triage, sla],
        model=make_provider(),
        mode="coordinate",
        hard=False,
        session_id=f"{SESSION}-soft",
    )

    note("soft Team coordinate starting")
    t0 = time.monotonic()
    team_result = await soft_team.arun(
        "INC-88421 AcmePay SEV page for BharatNova. "
        "Use tools if needed. Coordinate both specialists.\n\n"
        + ESCALATION_EMAIL
    )
    note(
        f"soft_team_ms={(time.monotonic()-t0)*1000:.0f} "
        f"tools={len(team_result.tool_activity or [])} "
        f"chars={len(team_result.output.text() or '')}"
    )
    if len(team_result.tool_activity or []) < 1:
        note("ISSUE: soft coordinate called zero tools — flaky delegation")

    # Agents as Workflow steps — prior Agent/Team output is the next input.
    drafter = Agent(
        model=make_provider(),
        role="Drafter",
        goal="Turn team findings into a short draft for the scribe",
        instructions=(
            "You receive the soft-team synthesis as your input. "
            "Write a compact draft covering severity, hypothesis, and actions."
        ),
        modalities="text",
        session_id=f"{SESSION}-draft",
    )
    scribe = Agent(
        model=make_provider(),
        role="Scribe",
        goal="Write approved packet after HITL",
        instructions=(
            "You receive the drafter's output as your input.\n"
            "1) write_file output/challenge_brief.md summarizing impact + actions.\n"
            "2) write_json output/challenge_packet.json as FinalPacket "
            "(severity SEV-*, approved=true, customer BharatNova).\n"
            "3) Final answer FinalPacket JSON only."
        ),
        tools=[FileTools(base_dir=str(work), json_schema=FinalPacket)],
        response_model=FinalPacket,
        require_tools=[
            "write_file:output/challenge_brief.md",
            "write_json:output/challenge_packet.json",
        ],
        max_tool_iterations=12,
        text_only=True,
        session_id=f"{SESSION}-scribe",
    )

    wf = (
        Workflow("challenge-hitl", session_id=SESSION, checkpointer=cp)
        .step("draft", drafter)
        .step("scribe", scribe, confirm=True)
    )

    note("workflow run — expect FlowPaused before scribe")
    paused = False
    try:
        await wf.arun(team_result.output)
        note("FAIL: expected FlowPaused")
    except FlowPaused as exc:
        paused = True
        note(f"HITL paused node={exc.pending.tool_name} thread={exc.thread_id}")
        assert exc.pending.tool_name == "scribe"

    assert paused, "confirm=True must raise FlowPaused"

    await wf.approve("scribe", status="approved")
    note("approved scribe — resuming")
    t1 = time.monotonic()
    final = await wf.arun(team_result.output, resume=True)
    note(
        f"resume_ms={(time.monotonic()-t1)*1000:.0f} "
        f"resumed={final.metadata.get('resumed')} "
        f"nudged={final.metadata.get('require_tools_nudged')} "
        f"missing={final.metadata.get('required_tools_missing')}"
    )

    packet = final.structured
    if not isinstance(packet, FinalPacket):
        packet = FinalPacket.model_validate_json(final.output.text() or "{}")

    brief = work / "output" / "challenge_brief.md"
    packet_path = work / "output" / "challenge_packet.json"
    assert brief.exists(), "require_tools must create challenge_brief.md"
    assert packet_path.exists(), "require_tools must create challenge_packet.json"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brief, OUTPUT / "challenge_brief.md")
    shutil.copy2(packet_path, OUTPUT / "challenge_packet.json")
    (OUTPUT / "challenge_obs.txt").write_text("\n".join(OBS) + "\n", encoding="utf-8")

    note(f"PASS packet severity={packet.severity} approved={packet.approved}")
    return packet


async def main() -> None:
    try:
        packet = await run_challenge()
        print("\n======== FINAL PACKET ========\n")
        print(packet.model_dump_json(indent=2))
        print("\n[ok] challenge 09 HITL + soft Team")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
