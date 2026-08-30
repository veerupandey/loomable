"""Team budget path must not wipe session_id across arun calls."""

from __future__ import annotations

import pytest

from loomable import Agent, Team
from loomable.agent import ModelSpec
from loomable.content import ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse


class EchoProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="team-ok")


@pytest.mark.asyncio
async def test_team_arun_preserves_session_with_max_delegations() -> None:
    member = Agent(
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        capabilities=ModelCapabilities(),
        role="Worker",
    )
    team = Team(
        [member],
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        session_id="team-session-1",
        max_delegations=2,
    )
    r1 = await team.arun("first")
    r2 = await team.arun("second")
    assert r1.session_id == "team-session-1"
    assert r2.session_id == "team-session-1"
    assert r1.session_id == r2.session_id
