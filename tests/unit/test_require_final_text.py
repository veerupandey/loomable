"""Unit tests for empty-final re-prompt and require_final_text."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.tools import tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall


class _EmptyThenNudgeProvider:
    """First call: tools; second call (with tools): empty; third (nudge): text."""

    def __init__(self) -> None:
        self.call_count = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        self.requests.append(request)
        if not request.tools:
            return ModelResponse(
                content="Confirmed: side effect done.",
                usage={"input_tokens": 2, "output_tokens": 4},
            )
        if self.call_count == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", tool_name="do_side_effect", args={"x": "1"})
                ],
                usage={"input_tokens": 5, "output_tokens": 2},
            )
        # Tools still advertised but model returns empty final.
        return ModelResponse(
            content="",
            usage={"input_tokens": 5, "output_tokens": 0},
        )


@tool
def do_side_effect(x: str) -> str:
    """Perform a side-effecting write."""
    return f"wrote:{x}"


@pytest.mark.asyncio
async def test_require_final_text_reprompts_after_empty_tool_final() -> None:
    provider = _EmptyThenNudgeProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[do_side_effect],
        require_final_text=True,
        max_tool_iterations=6,
    )

    result = await agent.arun("please do the side effect")

    assert (result.output.text() or "").strip() == "Confirmed: side effect done."
    assert result.metadata.get("final_text_reprompted") is True
    assert any(not r.tools for r in provider.requests)


@pytest.mark.asyncio
async def test_require_final_text_false_skips_nudge() -> None:
    provider = _EmptyThenNudgeProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[do_side_effect],
        require_final_text=False,
        max_tool_iterations=6,
    )

    result = await agent.arun("please do the side effect")

    text = (result.output.text() or "").strip()
    assert "Completed tool actions" in text
    assert "wrote:1" in text
    assert result.metadata.get("final_text_reprompted") is not True
    assert all(r.tools for r in provider.requests)
