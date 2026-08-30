"""Case mode guards for unsupported kwargs and from_agent fields."""

from __future__ import annotations

import pytest

from loomable import Agent
from loomable.agent import ModelSpec
from loomable.agent.errors import AgentConfigError
from loomable.case import Case
from loomable.content import ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse


class EchoProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


@pytest.mark.asyncio
async def test_case_mode_arun_rejects_images() -> None:
    agent = Agent(
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        capabilities=ModelCapabilities(),
        mode="case",
        goal="test goal",
    )
    with pytest.raises(AgentConfigError, match="images"):
        await agent.arun("hello", images=[])


def test_from_agent_rejects_subagents() -> None:
    member = Agent(
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        capabilities=ModelCapabilities(),
        role="R",
    )
    agent = Agent(
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        capabilities=ModelCapabilities(),
        mode="case",
        goal="g",
        subagents=[member],
    )
    with pytest.raises(AgentConfigError, match="subagents"):
        Case.from_agent(agent)
