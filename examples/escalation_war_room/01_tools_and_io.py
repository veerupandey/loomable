"""Phase 1a — Agent + domain tools + unstructured + structured I/O.

Tough real-world task: triage a Strategic-bank SEV email using ops tools,
then produce (1) a human war-room brief and (2) a typed EscalationPacket.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from loomable.agent import Agent, tool
from loomable.display import pp

from _common import ESCALATION_EMAIL, make_provider

# ---------------------------------------------------------------------------
# Fake but realistic ops systems (tools)
# ---------------------------------------------------------------------------


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up an AcmePay incident ticket by id (e.g. INC-88421)."""
    catalog = {
        "INC-88421": {
            "id": "INC-88421",
            "severity": "P1",
            "status": "Investigating",
            "service": "settlement-rail-v3",
            "region": "ap-south-1",
            "opened_at_ist": "2026-08-13T18:45:00+05:30",
            "reporter": "priya.nair@bharatnova.bank",
            "summary": "UPI settlement batches RETRYING/FAILED for BharatNova",
            "linked_change": "CHG-55219 cert rotation on connector pool",
        }
    }
    hit = catalog.get(ticket_id.upper())
    if not hit:
        return json.dumps({"error": f"ticket not found: {ticket_id}"})
    return json.dumps(hit, indent=2)


@tool
def get_customer_profile(customer_name: str) -> str:
    """Fetch partner commercial profile / tier for SLA decisions."""
    key = customer_name.strip().lower()
    if "bharatnova" in key or "bharat" in key:
        return json.dumps(
            {
                "customer": "BharatNova Bank",
                "tier": "Strategic",
                "arr_usd": 2_400_000,
                "products": ["settlement-rail-v3", "merchant-payouts"],
                "named_csm": "arjun.mehta@acmepay.io",
                "bridge_required_minutes": 30,
                "p1_response_minutes": 15,
            },
            indent=2,
        )
    return json.dumps({"error": f"unknown customer: {customer_name}"})


@tool
def get_service_health(service: str, region: str) -> str:
    """Return live-ish health signals for a service/region."""
    return json.dumps(
        {
            "service": service,
            "region": region,
            "overall": "degraded",
            "signals": {
                "settlement.batch.submit.error_rate": 0.37,
                "connector.pool.wait_p95_ms": 2400,
                "queue.depth": 42180,
                "last_successful_batch_ist": "2026-08-13T18:12:00+05:30",
            },
            "active_incidents": ["INC-88421"],
            "suspected_cause": "connector thread-pool saturation after CHG-55219",
        },
        indent=2,
    )


@tool
def calculate_sla_breach_minutes(
    opened_at_iso: str,
    tier: str,
    response_target_minutes: int = 15,
) -> str:
    """Compute minutes since open and whether response SLA is breached."""
    try:
        opened = datetime.fromisoformat(opened_at_iso)
    except ValueError:
        return json.dumps({"error": "invalid opened_at_iso"})
    now = datetime.now(timezone.utc)
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    elapsed = int((now - opened.astimezone(timezone.utc)).total_seconds() // 60)
    return json.dumps(
        {
            "tier": tier,
            "elapsed_minutes": elapsed,
            "response_target_minutes": response_target_minutes,
            "response_sla_breached": elapsed > response_target_minutes,
            "restore_target_minutes": 60 if tier.lower() == "strategic" else 120,
        },
        indent=2,
    )


@tool
def draft_customer_update(
    customer: str,
    severity: str,
    eta_minutes: int,
    mitigation: str,
) -> str:
    """Draft a customer-safe status update (no internal hostnames)."""
    return (
        f"Hi {customer} team — we are treating this as {severity}. "
        f"Mitigation in progress: {mitigation}. "
        f"Next update within {eta_minutes} minutes. "
        f"We will keep your bridge line open until settlement resumes."
    )


# ---------------------------------------------------------------------------
# Structured packet
# ---------------------------------------------------------------------------


class EscalationPacket(BaseModel):
    incident_id: str
    customer: str
    tier: Literal["Strategic", "Enterprise", "Growth", "Unknown"]
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    root_cause_hypothesis: str
    sla_response_breached: bool
    bridge_required: bool
    next_actions: list[str] = Field(default_factory=list)
    customer_update: str
    confidence: Literal["high", "medium", "low"]


ROLE = "Escalation Analyst"
GOAL = "Triage Strategic partner incidents into actionable war-room packets"
INSTRUCTIONS = """\
You are AcmePay's Escalation Analyst in a live war room.
ALWAYS use tools for ticket/customer/health/SLA facts — do not invent them.
Produce careful, operator-grade reasoning.
When asked for structured output, return ONLY JSON matching the schema keys.
Customer-facing text must be safe (no internal hosts, no blame).
"""


async def run_unstructured() -> None:
    agent = Agent(
        model=make_provider(),
        role=ROLE,
        goal=GOAL,
        instructions=INSTRUCTIONS,
        tools=[
            lookup_ticket,
            get_customer_profile,
            get_service_health,
            calculate_sla_breach_minutes,
            draft_customer_update,
        ],
    )
    result = await agent.arun(
        "War-room lead request:\n"
        "1) Triage the email below using tools.\n"
        "2) Write an UNSTRUCTURED brief for the bridge: impact, hypothesis, "
        "SLA status, next 3 actions, and a draft customer update.\n\n"
        f"{ESCALATION_EMAIL}"
    )
    print("\n======== UNSTRUCTURED BRIEF ========\n")
    pp(result)
    assert result.tool_activity, "expected tool use for triage"
    assert result.output.text().strip(), "empty brief"
    print(f"[ok] unstructured tools={len(result.tool_activity)}")


async def run_structured() -> None:
    agent = Agent(
        model=make_provider(),
        role=ROLE,
        goal=GOAL,
        instructions=INSTRUCTIONS
        + " Final answer MUST be raw JSON only with EscalationPacket keys.",
        tools=[
            lookup_ticket,
            get_customer_profile,
            get_service_health,
            calculate_sla_breach_minutes,
            draft_customer_update,
        ],
        response_model=EscalationPacket,
    )
    result = await agent.arun(
        "Using tools, produce an EscalationPacket JSON for this escalation.\n"
        "Example shape: "
        '{"incident_id":"INC-88421","customer":"BharatNova Bank","tier":"Strategic",'
        '"severity":"SEV-1","root_cause_hypothesis":"...","sla_response_breached":true,'
        '"bridge_required":true,"next_actions":["..."],"customer_update":"...",'
        '"confidence":"high"}\n\n'
        f"{ESCALATION_EMAIL}"
    )
    print("\n======== STRUCTURED PACKET ========\n")
    packet = result.structured
    assert isinstance(packet, EscalationPacket)
    print(packet.model_dump_json(indent=2))
    assert packet.incident_id.upper().startswith("INC")
    assert packet.tier == "Strategic"
    assert packet.severity in {"SEV-1", "SEV-2"}
    assert packet.next_actions
    print(f"[ok] structured tools={len(result.tool_activity or [])}")


async def main() -> None:
    await run_unstructured()
    await run_structured()


if __name__ == "__main__":
    asyncio.run(main())
