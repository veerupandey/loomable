"""STRESS EXAM — Full enterprise spine under pressure.

Combines in one run (what real SEV war rooms need):
  1) Workflow gather (tools + md/pdf/pptx)
  2) Kill after gather → resume (durability)
  3) Hard Team broadcast (triage + SLA parallel specialists)
  4) spawn_specialist cert auditor (ephemeral)
  5) Multimodal dashboard glance
  6) Scribe with write_json schema + response_model
  7) Session memory across resume

Observes UX/behavior into output/STRESS_OBSERVATIONS.md for framework fixes.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from loomable import (
    Agent,
    JsonFileCheckpointer,
    Step,
    Team,
    Workflow,
    spawn_specialist,
    tool,
)
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.persist.checkpoint import Checkpoint
from loomable.toolkits import FileTools, PDFTools, PPTTools

from _common import ESCALATION_EMAIL, FIXTURES, OUTPUT, ROOT, make_provider
from build_fixtures import main as build_fixtures

WORK = ROOT / "workspace_stress"
CKPT = ROOT / ".checkpoints_stress"
SESSION = "inc-88421-stress"
OBS: list[str] = []


def note(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    OBS.append(line)


# ---------------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------------


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
            "status": "Investigating",
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


@tool
def get_sla_clock(tier: str = "Strategic") -> str:
    """Return SLA clocks for partner tier."""
    return json.dumps(
        {
            "tier": tier,
            "ack_minutes": 15,
            "bridge_minutes": 30,
            "restore_minutes": 60,
            "elapsed_since_page_minutes": 23,
            "bridge_opened": False,
        }
    )


class EvidencePack(BaseModel):
    ticket_summary: str
    health_summary: str
    sla_summary: str
    doc_signals: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class SpecialistNotes(BaseModel):
    triage_view: str
    sla_view: str
    cert_audit: str
    visual_hint: str = ""


class FinalPacket(BaseModel):
    incident_id: str
    customer: str
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    root_hypothesis: str
    sla_status: str
    next_actions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    resumed: bool = False


def _prepare() -> Path:
    build_fixtures()
    if WORK.exists():
        shutil.rmtree(WORK)
    if CKPT.exists():
        shutil.rmtree(CKPT)
    (WORK / "fixtures").mkdir(parents=True)
    (WORK / "output").mkdir(parents=True)
    CKPT.mkdir(parents=True)
    for item in FIXTURES.iterdir():
        shutil.copy2(item, WORK / "fixtures" / item.name)
    return WORK


def _text_of(inp: Any) -> str:
    if hasattr(inp, "text") and callable(inp.text):
        return inp.text()
    if isinstance(inp, RunResult):
        return inp.output.text()
    return str(inp)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def make_gatherer(work: Path) -> Agent:
    return Agent(
        model=make_provider(),
        role="Evidence Gatherer",
        goal="Collect ticket/health/SLA/docs into EvidencePack",
        instructions=(
            "Use tools to gather facts then return EvidencePack JSON only.\n"
            "Must call: lookup_ticket INC-88421; get_service_health settlement-rail-v3 "
            "ap-south-1; get_sla_clock Strategic; list_directory fixtures; "
            f"read_file fixtures/runbook.md; read_pdf {work/'fixtures'/'strategic_sla.pdf'}; "
            f"read_pptx {work/'fixtures'/'incident_status.pptx'}.\n"
            "Do not write files."
        ),
        tools=[
            FileTools(base_dir=str(work)),
            PDFTools(),
            PPTTools(),
            lookup_ticket,
            get_service_health,
            get_sla_clock,
        ],
        response_model=EvidencePack,
        max_tool_iterations=18,
        modalities="text",
        session_id=f"{SESSION}-gather",
        memory_window=6,
        compaction_threshold=10,
    )


def make_team() -> Team:
    triage = Agent(
        model=make_provider(),
        role="Triage Lead",
        goal="Classify SEV and root hypothesis from evidence",
        instructions="Be terse. Output: severity + one hypothesis + 2 checks.",
        modalities="text",
    )
    sla = Agent(
        model=make_provider(),
        role="SLA Officer",
        goal="State SLA risk and bridge requirement",
        instructions="Be terse. Output: breach risk + bridge ETA urgency.",
        modalities="text",
    )
    return Team(
        members=[triage, sla],
        model=make_provider(),
        mode="broadcast",
        hard=True,
        session_id=f"{SESSION}-team",
    )


async def specialists_step(inp: Any, *, context=None) -> RunResult:
    """Hard Team broadcast + ephemeral cert auditor + optional image glance."""
    evidence = _text_of(inp)
    note(f"specialists_step input_chars={len(evidence)}")

    team = make_team()
    t0 = time.monotonic()
    team_result = await team.arun(
        "From this EvidencePack, give your specialist view:\n" + evidence[:6000]
    )
    note(f"team_broadcast_ms={(time.monotonic()-t0)*1000:.0f} hard={team_result.metadata.get('hard')}")

    t1 = time.monotonic()
    cert = await spawn_specialist(
        model=make_provider(),
        role="Cert Auditor",
        goal="Assess CHG-55219 cert rotation risk on connector pools",
        task=(
            "Given this war-room evidence, list 3 concrete checks for CHG-55219 "
            "cert rotation causing connector pool saturation. Be short.\n\n"
            + evidence[:3500]
        ),
        modalities="text",
    )
    note(f"spawn_specialist_ms={(time.monotonic()-t1)*1000:.0f} chars={len(cert)}")

    visual = ""
    dash = WORK / "fixtures" / "dashboard_spike.png"
    if dash.exists():
        try:
            t2 = time.monotonic()
            vision = Agent(
                model=make_provider(),
                role="Visual Analyst",
                instructions="One sentence: what the chart shows about failures vs success.",
                modalities="text+image",
            )
            vres = await vision.arun(
                "Glance this settlement error chart for BharatNova escalation.",
                images=[str(dash)],
            )
            visual = vres.output.text()
            note(f"vision_ms={(time.monotonic()-t2)*1000:.0f}")
        except Exception as exc:  # noqa: BLE001
            visual = f"(vision failed: {exc})"
            note(f"VISION_FAIL {exc}")

    notes = SpecialistNotes(
        triage_view=team_result.output.text()[:2000],
        sla_view="(included in broadcast)",
        cert_audit=cert[:2000],
        visual_hint=visual[:1000],
    )
    return RunResult(
        output=AgentOutput(parts=[Text(notes.model_dump_json(indent=2))]),
        session_id=SESSION,
        structured=notes,
        metadata={"team_hard": True},
    )


def make_scribe(work: Path, *, resumed: bool) -> Agent:
    return Agent(
        model=make_provider(),
        role="War-room Scribe",
        goal="Write brief + FinalPacket JSON",
        instructions=(
            "You receive EvidencePack JSON then SpecialistNotes JSON context.\n"
            "1) write_file output/stress_brief.md with impact, hypothesis, SLA, actions.\n"
            "2) write_json output/final_packet.json as FinalPacket "
            "(severity SEV-*, set resumed="
            + ("true" if resumed else "false")
            + ").\n"
            "3) Final answer MUST be FinalPacket JSON only."
        ),
        tools=[FileTools(base_dir=str(work), json_schema=FinalPacket)],
        response_model=FinalPacket,
        require_tools=["write_file", "write_json"],
        max_tool_iterations=12,
        text_only=True,
        session_id=f"{SESSION}-scribe",
    )


async def run_stress() -> FinalPacket:
    work = _prepare()
    cp = JsonFileCheckpointer(str(CKPT))
    note("fixtures ready; starting gather")

    gatherer = make_gatherer(work)

    # ---- Phase 1: gather only, then kill ----
    wf_gather = Workflow(
        "stress-gather",
        session_id=SESSION,
        checkpointer=cp,
        memory=True,
    ).step("gather", gatherer)

    t0 = time.monotonic()
    gather_result = await wf_gather.arun(ESCALATION_EMAIL)
    note(f"gather_ms={(time.monotonic()-t0)*1000:.0f} tools={len(gather_result.tool_activity or [])}")
    pack = gather_result.structured
    if not isinstance(pack, EvidencePack):
        # Fallback: try parse text
        note(f"GATHER_STRUCTURED_MISS type={type(pack)} — parsing text")
        pack = EvidencePack.model_validate_json(gather_result.output.text())

    # Simulate crash: incomplete checkpoint with gather done
    from loomable.flow.state import SharedState

    state = SharedState()
    state.write("gather", gather_result.output)
    # Also stash specialists placeholder not run yet
    await cp.put(
        Checkpoint(
            thread_id=SESSION,
            step=1,
            session_state={
                "shared_state": state.snapshot(),
                "completed_node_ids": ["gather"],
            },
            complete=False,
        )
    )
    note("KILL simulated after gather — incomplete checkpoint written")

    # ---- Phase 2: resume → specialists → scribe ----
    # Build a workflow that includes gather (skipped) + specialists + scribe
    calls = {"gather": 0}

    async def gather_guard(inp, *, context=None):
        calls["gather"] += 1
        note("ERROR gather re-executed after resume!")
        return await gatherer.arun(inp)

    async def specialists_wrap(inp, *, context=None):
        # Prefer restored gather output
        return await specialists_step(inp, context=context)

    scribe = make_scribe(work, resumed=True)

    async def scribe_wrap(inp, *, context=None):
        # Combine evidence + specialist notes for scribe
        evidence = pack.model_dump_json(indent=2)
        notes = _text_of(inp)
        prompt = (
            "EvidencePack:\n"
            + evidence
            + "\n\nSpecialistNotes:\n"
            + notes
            + "\n\nProduce FinalPacket. Set resumed=true."
        )
        return await scribe.arun(prompt)

    wf2 = (
        Workflow("stress-full", session_id=SESSION, checkpointer=cp, memory=True)
        .step("gather", gather_guard)
        .step("specialists", specialists_wrap)
        .step("scribe", scribe_wrap)
    )

    t1 = time.monotonic()
    final = await wf2.arun(ESCALATION_EMAIL, resume=True)
    note(
        f"resume_full_ms={(time.monotonic()-t1)*1000:.0f} "
        f"resumed={final.metadata.get('resumed')} "
        f"skipped={final.metadata.get('skipped_nodes')} "
        f"gather_recalls={calls['gather']}"
    )

    if calls["gather"] != 0:
        note("FAIL: gather was re-run after resume")

    packet = final.structured
    if not isinstance(packet, FinalPacket):
        note(f"SCRIBE_STRUCTURED_MISS type={type(packet)} — parsing")
        packet = FinalPacket.model_validate_json(final.output.text())

    brief = work / "output" / "stress_brief.md"
    packet_path = work / "output" / "final_packet.json"
    assert brief.exists(), "scribe must write_file stress_brief.md (require_tools)"
    assert packet_path.exists(), "scribe must write_json final_packet.json (require_tools)"
    note(
        f"scribe_artifacts ok nudged={final.metadata.get('require_tools_nudged')} "
        f"missing={final.metadata.get('required_tools_missing')}"
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brief, OUTPUT / "stress_brief.md")
    shutil.copy2(packet_path, OUTPUT / "final_packet.json")

    # Assertions
    assert calls["gather"] == 0, "resume must skip gather"
    assert final.metadata.get("resumed") is True
    assert packet.severity.startswith("SEV-")
    assert packet.next_actions
    assert packet.incident_id.upper().startswith("INC")
    assert not final.metadata.get("required_tools_missing"), final.metadata
    note(f"FINAL severity={packet.severity} confidence={packet.confidence} resumed={packet.resumed}")
    return packet


def write_observations(exc: BaseException | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "STRESS_OBSERVATIONS.md"
    body = ["# Stress exam observations\n", "\n## Timeline\n"]
    body.extend(f"- {line}\n" for line in OBS)
    if exc is not None:
        body.append("\n## Exception\n```\n")
        body.append("".join(traceback.format_exception(exc)))
        body.append("```\n")
    path.write_text("".join(body), encoding="utf-8")
    note(f"wrote {path}")


async def main() -> None:
    try:
        packet = await run_stress()
        print("\n======== FINAL PACKET ========\n")
        print(packet.model_dump_json(indent=2))
        write_observations()
        print("[ok] STRESS EXAM PASSED")
    except Exception as exc:
        note(f"STRESS_FAIL {exc}")
        write_observations(exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
