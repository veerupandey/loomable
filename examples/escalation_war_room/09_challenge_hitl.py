"""CHALLENGE 09 — HITL confirm + soft Team + require_tools under stress.

Tougher than 08:
  1) Soft Team ``coordinate`` (LLM must actually delegate)
  2) Workflow.step(..., confirm=True) HITL pause before scribe
  3) approve() + resume
  4) Scribe with require_tools write_file/write_json (no synthesis fallback)
  5) Notes behavior for soft-team flakiness / HITL ergonomics

Fails hard if artifacts missing or HITL/resume broken.
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

from loomable import Agent, JsonFileCheckpointer, Team, Workflow, tool, FlowPaused
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
        mode="coordinate",  # soft — LLM must delegate
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

    async def draft_step(inp, *, context=None):
        text = inp.text() if hasattr(inp, "text") and callable(inp.text) else str(inp)
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text

        return RunResult(
            output=AgentOutput(parts=[Text("DRAFT:\n" + text[:8000])]),
            session_id=SESSION,
        )

    scribe = Agent(
        model=make_provider(),
        role="Scribe",
        goal="Write approved packet after HITL",
        instructions=(
            "1) write_file output/challenge_brief.md summarizing impact + actions.\n"
            "2) write_json output/challenge_packet.json as FinalPacket "
            "(severity SEV-*, approved=true, customer BharatNova).\n"
            "3) Final answer FinalPacket JSON only."
        ),
        tools=[FileTools(base_dir=str(work), json_schema=FinalPacket)],
        response_model=FinalPacket,
        require_tools=["write_file", "write_json"],
        max_tool_iterations=12,
        text_only=True,
        session_id=f"{SESSION}-scribe",
    )

    async def scribe_step(inp, *, context=None):
        text = inp.text() if hasattr(inp, "text") and callable(inp.text) else str(inp)
        return await scribe.arun(
            "After human approval, produce FinalPacket from this draft:\n" + text
        )

    wf = (
        Workflow("challenge-hitl", session_id=SESSION, checkpointer=cp)
        .step("draft", draft_step)
        .step("scribe", scribe_step, confirm=True)
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

    brief = work / "output" / "challenge_brief.md"
    packet_path = work / "output" / "challenge_packet.json"
    assert brief.exists(), "require_tools must force write_file"
    assert packet_path.exists(), "require_tools must force write_json"
    assert not final.metadata.get("required_tools_missing")

    packet = final.structured
    if not isinstance(packet, FinalPacket):
        packet = FinalPacket.model_validate_json(final.output.text())
    assert packet.severity.startswith("SEV-")
    assert packet.next_actions
    assert packet.approved is True

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brief, OUTPUT / "challenge_brief.md")
    shutil.copy2(packet_path, OUTPUT / "challenge_packet.json")
    note(f"FINAL severity={packet.severity} confidence={packet.confidence}")
    return packet


def write_obs(exc: BaseException | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "CHALLENGE_OBSERVATIONS.md"
    body = ["# Challenge 09 observations\n", "\n## Timeline\n"]
    body.extend(f"- {line}\n" for line in OBS)
    if exc is not None:
        body.append("\n## Exception\n```\n")
        body.append("".join(traceback.format_exception(exc)))
        body.append("```\n")
    path.write_text("".join(body), encoding="utf-8")


async def main() -> None:
    try:
        packet = await run_challenge()
        print("\n======== CHALLENGE PACKET ========\n")
        print(packet.model_dump_json(indent=2))
        write_obs()
        print("[ok] CHALLENGE 09 PASSED")
    except Exception as exc:
        note(f"CHALLENGE_FAIL {exc}")
        write_obs(exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
