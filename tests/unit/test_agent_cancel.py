"""BuiltAgent.cancel / Agent.cancel cooperative cancellation."""

from __future__ import annotations

import asyncio

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.context import RunContext, StopReason
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall


class _NeverFinalProvider:
    """Keeps requesting tools so cancel is observed at a loop boundary."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(0.01)
        return ModelResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=str(self.calls),
                    tool_name="think",
                    args={"thought": f"step-{self.calls}"},
                )
            ],
        )


@pytest.mark.asyncio
async def test_built_agent_cancel_marks_active_context() -> None:
    from loomable.agent.reasoning import make_think_tool

    provider = _NeverFinalProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[make_think_tool()],
        think_tool=False,
        max_tool_iterations=20,
        use_llm_summarizer=False,
    )
    built = agent.build()
    ctx = RunContext(max_steps=20)

    async def _run():
        return await built.arun("go", context=ctx)

    task = asyncio.create_task(_run())
    # Wait until the run has attached the context and started looping
    for _ in range(100):
        if built._active_ctx is ctx and provider.calls >= 1:
            break
        await asyncio.sleep(0.01)
    assert built._active_ctx is ctx
    assert built.cancel() is True
    assert ctx.cancelled is True
    result = await task
    assert (result.metadata or {}).get("stop_reason") == StopReason.CANCELLED
    assert provider.calls < 20


@pytest.mark.asyncio
async def test_agent_cancel_forwards_to_built() -> None:
    provider = _NeverFinalProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        use_llm_summarizer=False,
    )
    assert agent.cancel() is False  # nothing built yet
    built = agent._get_built()
    built._active_ctx = RunContext()
    assert agent.cancel() is True
    assert built._active_ctx.cancelled is True
