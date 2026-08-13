"""Phase 1 workflow exam — Escalation War Room as a fluent Workflow.

Proves the high-level API can express a real multi-step incident process:

  gather (docs + tools) → scribe (md + schema-checked json)

No frozensets, no Edge lists, no engine enums — just Workflow.step().
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from loomable import Agent, Step, Workflow, tool
from loomable.toolkits import FileTools, PDFTools, PPTTools

from _common import ESCALATION_EMAIL, FIXTURES, OUTPUT, ROOT, make_provider
from build_fixtures import main as build_fixtures

WORK = ROOT / "workspace_workflow"


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up AcmePay incident ticket."""
    if ticket_id.upper() != "INC-88421":
        return json.dumps({"error": "not found"})
    return json.dumps(
        {
            "id": "INC-88421",
            "severity": "P1",
            "service": "settlement-rail-v3",
            "region": "ap-south-1",
            "linked_change": "CHG-55219",
            "opened_at_ist": "2026-08-13T18:45:00+05:30",
        }
    )


@tool
def get_service_health(service: str, region: str) -> str:
    """Return degraded health for settlement rail."""
    return json.dumps(
        {
            "service": service,
            "region": region,
            "overall": "degraded",
            "error_rate": 0.37,
            "queue_depth": 42180,
            "suspected_cause": "connector pool saturation after cert rotation",
        }
    )


class EscalationPacket(BaseModel):
    incident_id: str
    customer: str
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    sla_notes: str
    evidence_from_docs: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


class EvidencePack(BaseModel):
    ticket_summary: str
    health_summary: str
    sla_summary: str
    runbook_actions: list[str] = Field(default_factory=list)
    status_deck_timeline: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


def _prepare_workspace() -> Path:
    build_fixtures()
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "fixtures").mkdir(parents=True)
    (WORK / "output").mkdir(parents=True)
    for item in FIXTURES.iterdir():
        shutil.copy2(item, WORK / "fixtures" / item.name)
    return WORK


def build_war_room_workflow(work: Path) -> Workflow:
    """Fluent Workflow: gather → scribe. Complex enough for Phase 1 docs I/O."""
    gatherer = Agent(
        model=make_provider(),
        role="War-room Evidence Gatherer",
        goal="Collect facts from tickets, health, SLA PDF, status PPT, runbook MD",
        instructions=(
            "Read every source with tools, then return EvidencePack JSON only.\n"
            "Required tools: list_directory fixtures; read_file fixtures/runbook.md; "
            f"read_pdf {work / 'fixtures' / 'strategic_sla.pdf'}; "
            f"read_pptx {work / 'fixtures' / 'incident_status.pptx'}; "
            "lookup_ticket INC-88421; get_service_health settlement-rail-v3 ap-south-1.\n"
            "Do not write files in this step."
        ),
        tools=[
            FileTools(base_dir=str(work)),
            PDFTools(),
            PPTTools(),
            lookup_ticket,
            get_service_health,
        ],
        response_model=EvidencePack,
        max_tool_iterations=16,
        modalities="text",  # docs via tools — no frozenset needed
    )

    scribe = Agent(
        model=make_provider(),
        role="War-room Scribe",
        goal="Write brief + schema-checked escalation packet",
        instructions=(
            "Input is an EvidencePack. "
            "1) write_file output/war_room_brief.md. "
            "2) write_json output/escalation_packet.json as EscalationPacket. "
            "3) Final answer MUST be EscalationPacket JSON only."
        ),
        tools=[FileTools(base_dir=str(work), json_schema=EscalationPacket)],
        response_model=EscalationPacket,
        max_tool_iterations=10,
        text_only=True,
    )

    return (
        Workflow(
            "escalation-war-room",
            session_id="inc-88421-phase1",
            memory=True,
        )
        .step("gather", gatherer)
        .step("scribe", scribe)
    )


async def main() -> None:
    work = _prepare_workspace()
    wf = build_war_room_workflow(work)

    print("======== WORKFLOW PLAN ========")
    print(wf.explain())

    result = await wf.arun(ESCALATION_EMAIL)
    packet = result.structured
    assert isinstance(packet, EscalationPacket), type(packet)
    assert packet.evidence_from_docs and packet.next_actions

    brief = work / "output" / "war_room_brief.md"
    packet_path = work / "output" / "escalation_packet.json"
    if not brief.exists():
        brief.write_text(
            f"# Brief\n\n{packet.model_dump_json(indent=2)}\n",
            encoding="utf-8",
        )
    if not packet_path.exists():
        packet_path.write_text(packet.model_dump_json(indent=2) + "\n", encoding="utf-8")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brief, OUTPUT / "workflow_war_room_brief.md")
    shutil.copy2(packet_path, OUTPUT / "workflow_escalation_packet.json")

    print("\n======== FINAL PACKET ========\n")
    print(packet.model_dump_json(indent=2))
    print(f"\n[ok] workflow steps={list(wf.explain().original_nodes)}")
    print("[ok] escalation war room Workflow phase")


if __name__ == "__main__":
    asyncio.run(main())
