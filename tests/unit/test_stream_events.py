"""Unit tests for AG-UI stream protocol and Agent.astream_events."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec, tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.stream import (
    RUN_FINISHED,
    RUN_STARTED,
    TEXT_MESSAGE_CONTENT,
    TOOL_CALL_START,
    StreamEvent,
    sse_encode,
)


def test_sse_encode_framing() -> None:
    ev = StreamEvent(type=RUN_STARTED, run_id="r1", session_id="s1", data={"ok": True})
    frame = sse_encode(ev).decode("utf-8")
    assert frame.startswith("event: RUN_STARTED\n")
    assert "data: {" in frame
    assert frame.endswith("\n\n")


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="hello world", usage={"input_tokens": 1, "output_tokens": 2})


class _ToolThenText:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="ping", args={"x": "1"})],
                usage={"input_tokens": 1, "output_tokens": 1},
            )
        return ModelResponse(content="pong-done", usage={"input_tokens": 1, "output_tokens": 2})


@tool
def ping(x: str) -> str:
    """Ping tool."""
    return f"pong:{x}"


@pytest.mark.asyncio
async def test_agent_astream_events_lifecycle_and_text() -> None:
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Echo()),
        modalities="text",
    )
    types: list[str] = []
    async for ev in agent.astream_events("hi"):
        types.append(ev.type)
    assert RUN_STARTED in types
    assert TEXT_MESSAGE_CONTENT in types
    assert RUN_FINISHED in types


@pytest.mark.asyncio
async def test_agent_astream_events_tool_loop() -> None:
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_ToolThenText()),
        tools=[ping],
        modalities="text",
        max_tool_iterations=6,
    )
    types: list[str] = []
    async for ev in agent.astream_events("ping please"):
        types.append(ev.type)
    assert RUN_STARTED in types
    assert TOOL_CALL_START in types
    assert TEXT_MESSAGE_CONTENT in types
    assert RUN_FINISHED in types
