"""Phase D gate — Team hard modes + spawn_specialist.

Runs a deterministic broadcast Team (no LLM coordinator required for fan-out)
and an ephemeral specialist spawn for cert audit commentary.
"""

from __future__ import annotations

import asyncio

from loomable import Agent, Team, spawn_specialist
from loomable.agent import ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse

from _common import ESCALATION_EMAIL, make_provider


class _LabelEcho:
    def __init__(self, label: str) -> None:
        self.label = label

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=f"{self.label}-done")


async def main() -> None:
    # --- Hard broadcast (deterministic, no coordinator tool luck) ---
    triage = Agent(
        model=ModelSpec(provider="t", provider_impl=_LabelEcho("triage")),
        role="Triage",
        goal="Classify severity",
    )
    comms = Agent(
        model=ModelSpec(provider="c", provider_impl=_LabelEcho("comms")),
        role="Comms",
        goal="Draft customer update",
    )
    team = Team(
        members=[triage, comms],
        model=ModelSpec(provider="m", provider_impl=_LabelEcho("mgr")),
        mode="broadcast",
        hard=True,
    )
    result = await team.arun("Assess INC-88421 settlement spike")
    text = result.output.text()
    assert "Triage" in text and "Comms" in text
    assert "triage-done" in text and "comms-done" in text
    assert result.metadata.get("hard") is True
    print("[ok] hard broadcast team")

    # --- Hard sequential pipeline ---
    seq = Team(
        members=[triage, comms],
        model=ModelSpec(provider="m", provider_impl=_LabelEcho("mgr")),
        mode="sequential",
        hard=True,
    )
    seq_result = await seq.arun("Pipeline the incident")
    assert "comms-done" in seq_result.output.text()
    print("[ok] hard sequential team")

    # --- Ephemeral spawn (live Gemini if key present, else echo) ---
    try:
        provider = make_provider()
        spawned = await spawn_specialist(
            model=provider,
            role="Cert Auditor",
            goal="Review change risk for connector pools",
            task=(
                "In one short paragraph, what should we check in CHG-55219 "
                "after a cert rotation that may saturate connector pools?\n\n"
                + ESCALATION_EMAIL[:400]
            ),
            modalities="text",
        )
        assert len(spawned.strip()) > 20
        print("[ok] spawn_specialist (live)")
        print(spawned[:300])
    except Exception as exc:  # noqa: BLE001
        # Offline fallback
        spawned = await spawn_specialist(
            model=ModelSpec(provider="e", provider_impl=_LabelEcho("auditor")),
            role="Cert Auditor",
            task="Review CHG-55219",
        )
        assert "auditor-done" in spawned
        print(f"[ok] spawn_specialist (echo fallback: {exc})")

    print("[ok] Phase D team/spawn gate")


if __name__ == "__main__":
    asyncio.run(main())
