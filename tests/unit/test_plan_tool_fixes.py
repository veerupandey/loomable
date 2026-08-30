"""plan_tool synthesizer + exclude_tools in plan workers."""

from __future__ import annotations

import pytest

from loomable import Agent, tool
from loomable.agent import ModelSpec
from loomable.agent.reasoning import make_plan_tool
from loomable.content import AgentInput, ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse


class EchoProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


@pytest.mark.asyncio
async def test_run_tool_loop_honors_exclude_tools() -> None:
    provider = EchoProvider()

    @tool
    def alpha() -> str:
        return "a"

    @tool
    def plan(task: str) -> str:
        return task

    agent = Agent(
        model=ModelSpec(provider="p", provider_impl=provider),
        capabilities=ModelCapabilities(),
        tools=[alpha, plan],
    )
    built = agent.build()
    seen: list[list[str]] = []

    original_collect = None

    async def _run_and_capture() -> None:
        nonlocal original_collect
        # Invoke tool loop once; provider receives advertised schema names.
        calls_before = len(seen)

        class CaptureProvider(EchoProvider):
            async def complete(self, request: ModelRequest) -> ModelResponse:
                names = [
                    t.get("function", {}).get("name", "")
                    for t in (request.tools or [])
                    if isinstance(t, dict)
                ]
                seen.append([n for n in names if n])
                return ModelResponse(content="done")

        built.model_interface._providers["p"] = CaptureProvider()
        await built._run_tool_loop(
            AgentInput.from_text("work"),
            include_history=False,
            exclude_tools=frozenset({"plan"}),
        )

    await _run_and_capture()
    assert seen
    assert "plan" not in seen[0]
    assert "alpha" in seen[0]


def test_make_plan_tool_is_not_idempotent() -> None:
    built = Agent(
        model=ModelSpec(provider="p", provider_impl=EchoProvider()),
        capabilities=ModelCapabilities(),
    ).build()
    plan_tool = make_plan_tool(built)
    assert plan_tool.idempotent is False
