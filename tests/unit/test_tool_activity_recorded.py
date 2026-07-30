# Feature: agent-ergonomics, Property 7
"""Property 7: Tool activity is recorded.

For any run that executes tool calls, the returned RunResult.tool_activity
contains one entry per executed tool call. Each entry records the tool name
(via call_id correlation), args, and result.

**Validates: Requirements 3.4**
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fakes (reusing patterns from test_tool_loop_execution.py)
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """A model provider that returns a scripted sequence of responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._call_index >= len(self._responses):
            return ModelResponse(content="fallback", tool_calls=[])
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


class RecordingTool(Tool):
    """A tool that records invocations and returns a fixed result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Tool {name}"
        self.invocations: list[dict[str, Any]] = []

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(args))
        return ToolResult(content=f"result-from-{self.name}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolActivityRecorded:
    """Property 7: Tool activity is recorded."""

    @pytest.mark.asyncio
    async def test_single_tool_call_recorded_in_activity(self) -> None:
        """A single tool call produces exactly one entry in tool_activity."""
        call_id = str(uuid.uuid4())
        tool_name = "calculator"
        tool_args = {"x": 1, "y": 2}

        responses = [
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id=call_id, tool_name=tool_name, args=tool_args)],
            ),
            ModelResponse(content="done", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        recording_tool = RecordingTool("calculator")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[recording_tool],
        )
        built = agent.build()

        result = await built.arun("compute something")

        assert isinstance(result, RunResult)
        assert len(result.tool_activity) == 1

        outcome = result.tool_activity[0]
        # The call_id links back to the original tool call
        assert outcome.call_id == call_id
        # The result carries the tool's return value
        assert outcome.result is not None
        assert outcome.result.content == "result-from-calculator"
        assert outcome.error is None

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_iteration(self) -> None:
        """Multiple tool calls in a single model response each produce an entry."""
        call_ids = [str(uuid.uuid4()) for _ in range(3)]
        tool_names = ["tool_a", "tool_b", "tool_c"]
        tool_args_list = [{"key": "a"}, {"key": "b"}, {"key": "c"}]

        tool_calls = [
            ToolCall(id=call_ids[i], tool_name=tool_names[i], args=tool_args_list[i])
            for i in range(3)
        ]

        responses = [
            ModelResponse(content="", tool_calls=tool_calls),
            ModelResponse(content="all done", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        tools = [RecordingTool(name) for name in tool_names]

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=tools,
        )
        built = agent.build()

        result = await built.arun("do things")

        assert len(result.tool_activity) == 3

        # Each outcome corresponds to the correct tool call
        for i, outcome in enumerate(result.tool_activity):
            assert outcome.call_id == call_ids[i]
            assert outcome.result is not None
            assert outcome.result.content == f"result-from-{tool_names[i]}"

    @pytest.mark.asyncio
    async def test_tool_calls_across_multiple_iterations(self) -> None:
        """Tool calls across multiple loop iterations all appear in tool_activity."""
        # Iteration 1: one tool call
        call_id_1 = str(uuid.uuid4())
        # Iteration 2: two tool calls
        call_id_2a = str(uuid.uuid4())
        call_id_2b = str(uuid.uuid4())

        responses = [
            # Iteration 1
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id=call_id_1, tool_name="fetcher", args={"url": "http://a.com"}),
                ],
            ),
            # Iteration 2
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id=call_id_2a, tool_name="fetcher", args={"url": "http://b.com"}),
                    ToolCall(id=call_id_2b, tool_name="parser", args={"fmt": "json"}),
                ],
            ),
            # Final answer
            ModelResponse(content="final", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        fetcher = RecordingTool("fetcher")
        parser = RecordingTool("parser")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[fetcher, parser],
        )
        built = agent.build()

        result = await built.arun("process data")

        # Total: 3 tool calls across 2 iterations
        assert len(result.tool_activity) == 3

        recorded_call_ids = [o.call_id for o in result.tool_activity]
        assert call_id_1 in recorded_call_ids
        assert call_id_2a in recorded_call_ids
        assert call_id_2b in recorded_call_ids

        # All outcomes have results (no errors)
        for outcome in result.tool_activity:
            assert outcome.result is not None
            assert outcome.error is None

    @pytest.mark.asyncio
    async def test_no_tool_calls_means_empty_activity(self) -> None:
        """When the model returns no tool calls, tool_activity is empty."""
        responses = [
            ModelResponse(content="just an answer", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        recording_tool = RecordingTool("unused_tool")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[recording_tool],
        )
        built = agent.build()

        result = await built.arun("question")

        assert result.tool_activity == []

    @pytest.mark.asyncio
    async def test_tool_activity_result_content_matches_tool_output(self) -> None:
        """Each tool_activity entry's result matches what the tool actually returned."""
        call_id = str(uuid.uuid4())

        responses = [
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id=call_id, tool_name="greeter", args={"name": "world"}),
                ],
            ),
            ModelResponse(content="greeting sent", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        greeter = RecordingTool("greeter")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[greeter],
        )
        built = agent.build()

        result = await built.arun("greet someone")

        assert len(result.tool_activity) == 1
        outcome = result.tool_activity[0]
        assert outcome.call_id == call_id
        assert outcome.result.content == "result-from-greeter"
        # Verify the tool was actually invoked with the correct args
        assert greeter.invocations == [{"name": "world"}]
