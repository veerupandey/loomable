"""Evaluate two incident handoffs with a real provider."""
import asyncio
import json

import pytest

from examples.incident_triage.__main__ import FIXTURES, triage
from tests.integration.test_model_live import live_provider

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "fixture,severity,owner,mitigation,threshold",
    [
        ("payments", "SEV2", "Payments On-Call", "rollback", "2%"),
        ("search", "SEV3", "Search On-Call", "throttl", "250"),
    ],
)
async def test_incident_handoff_matches_runbook(fixture, severity, owner, mitigation, threshold):
    packet = json.loads((FIXTURES / f"{fixture}.json").read_text())
    async with live_provider() as provider, asyncio.timeout(75):
        result = await triage(provider, packet)

    assert result.severity == severity
    assert result.owner.casefold() == owner.casefold()
    assert mitigation in result.proposed_mitigation.casefold().replace("roll back", "rollback")
    assert threshold in result.verification
    assert "M1" in result.evidence_ids and result.runbook_id == "R1"
