"""Phase 1b — PDF / PPT / Markdown input + markdown/json output.

Real war-room pattern (two agent turns, one exam):
  1) Evidence gatherer reads md/pdf/pptx + ops tools → evidence pack (text)
  2) Scribe writes war_room_brief.md + escalation_packet.json from evidence

This also surfaces ISSUE-WR-001: gather-only turns often end with empty text
unless we force a textual pack / structured recovery.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from loomable.agent import Agent, tool
from loomable.toolkits import FileTools, PDFTools, PPTTools

from _common import ESCALATION_EMAIL, FIXTURES, OUTPUT, ROOT, make_provider
from build_fixtures import main as build_fixtures

WORK = ROOT / "workspace"


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
    """Normalized evidence extracted from tools + documents."""

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


async def gather_evidence(work: Path) -> EvidencePack:
    agent = Agent(
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
    )
    result = await agent.arun(
        "Build an EvidencePack for this escalation.\n\n" + ESCALATION_EMAIL
    )
    pack = result.structured
    assert isinstance(pack, EvidencePack)
    print(f"[ok] evidence tools={len(result.tool_activity or [])}")
    print(pack.model_dump_json(indent=2)[:1200])
    return pack


async def write_artifacts(work: Path, pack: EvidencePack) -> EscalationPacket:
    """Scribe writes the human brief via tool; packet comes from response_model.

    Splitting channels avoids ISSUE-WR-003 (unchecked JSON on disk) and makes the
    machine packet reliable under Gemini tool-only empty finals.
    """

    agent = Agent(
        model=make_provider(),
        role="War-room Scribe",
        goal="Write customer-safe brief file and return EscalationPacket JSON",
        instructions=(
            "1) write_file output/war_room_brief.md using the evidence "
            "(impact, hypothesis, SLA, next actions, draft update).\n"
            "2) Final answer MUST be EscalationPacket JSON only with keys "
            "incident_id, customer, severity (SEV-*), sla_notes, "
            "evidence_from_docs, next_actions, confidence."
        ),
        tools=[FileTools(base_dir=str(work))],
        response_model=EscalationPacket,
        max_tool_iterations=8,
    )
    result = await agent.arun(
        "Create war-room outputs from this EvidencePack JSON:\n"
        + pack.model_dump_json(indent=2)
    )
    print(
        f"[scribe tools={len(result.tool_activity or [])}] "
        f"text_chars={len(result.output.text() or '')}"
    )
    packet = result.structured
    assert isinstance(packet, EscalationPacket)

    brief = work / "output" / "war_room_brief.md"
    if not brief.exists():
        # Fallback: synthesize a brief from the packet if the model skipped write_file
        print("[issue] ISSUE-WR-001/003 scribe skipped write_file; synthesizing brief")
        brief.write_text(
            "# War-room brief (synthesized)\n\n"
            f"- Incident: {packet.incident_id}\n"
            f"- Customer: {packet.customer}\n"
            f"- Severity: {packet.severity}\n"
            f"- SLA: {packet.sla_notes}\n\n"
            "## Evidence\n"
            + "\n".join(f"- {e}" for e in packet.evidence_from_docs)
            + "\n\n## Next actions\n"
            + "\n".join(f"- {a}" for a in packet.next_actions)
            + "\n",
            encoding="utf-8",
        )

    packet_path = work / "output" / "escalation_packet.json"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brief, OUTPUT / "war_room_brief.md")
    shutil.copy2(packet_path, OUTPUT / "escalation_packet.json")
    print("\n======== war_room_brief.md (head) ========\n")
    print(brief.read_text(encoding="utf-8")[:1600])
    print("\n======== escalation_packet.json ========\n")
    print(packet.model_dump_json(indent=2))
    return packet


async def main() -> None:
    work = _prepare_workspace()
    pack = await gather_evidence(work)
    packet = await write_artifacts(work, pack)
    assert packet.evidence_from_docs and packet.next_actions
    print("[ok] document I/O phase")


if __name__ == "__main__":
    asyncio.run(main())
