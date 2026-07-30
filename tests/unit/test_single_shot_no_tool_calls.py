# Feature: agent-ergonomics, Property 8
"""Property 8: No tool calls means single-shot.

For any model response with no tool calls, the agent makes exactly one model
call and returns its output (no extra iterations). This holds both when the
agent has tools registered (but the model doesn't call them) and when no tools
are registered at all.

**Validates: Requirements 3.6**
"""

from __future__ import annotations

from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fakes (reusing ScriptedProvider / RecordingTool patterns)
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


class TestSingleShotNoToolCalls:
    """Property 8: No tool calls means single-shot."""

    @pytest.mark.asyncio
    async def test_no_tools_registered_single_model_call(self) -> None:
        """When no tools are registered, the agent uses the single-shot path
        and makes exactly one model call."""
        responses = [
            ModelResponse(content="hello world", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            # No tools registered
        )
        built = agent.build()

        result = await built.arun("say hello")

        assert isinstance(result, RunResult)
        # Exactly one model call was made
        assert len(provider.requests) == 1
        # The output matches the model response
        assert result.output.text() == "hello world"
        # No tool activity
        assert result.tool_activity == []

    @pytest.mark.asyncio
    async def test_tools_registered_but_model_returns_no_tool_calls(self) -> None:
        """When tools are registered but the model doesn't call any, the agent
        makes exactly one model call and returns immediately (no extra iterations)."""
        responses = [
            ModelResponse(content="I can answer without tools", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        tool = RecordingTool("search")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[tool],
        )
        built = agent.build()

        result = await built.arun("what is 2+2?")

        assert isinstance(result, RunResult)
        # Exactly one model call was made despite tools being available
        assert len(provider.requests) == 1
        # The output matches the model's direct response
        assert result.output.text() == "I can answer without tools"
        # No tool activity since model didn't call any tools
        assert result.tool_activity == []
        # The tool was never invoked
        assert tool.invocations == []

    @pytest.mark.asyncio
    async def test_multiple_tools_registered_no_calls_made(self) -> None:
        """Even with multiple tools registered, if the model returns no tool
        calls, exactly one model call is made."""
        responses = [
            ModelResponse(content="direct answer", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        tools = [
            RecordingTool("calculator"),
            RecordingTool("search"),
            RecordingTool("weather"),
        ]

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=tools,
        )
        built = agent.build()

        result = await built.arun("tell me a joke")

        assert isinstance(result, RunResult)
        # Exactly one model call
        assert len(provider.requests) == 1
        # Output is the model's direct response
        assert result.output.text() == "direct answer"
        # No tool activity
        assert result.tool_activity == []
        # None of the tools were invoked
        for tool in tools:
            assert tool.invocations == []

    @pytest.mark.asyncio
    async def test_tool_schemas_advertised_in_request(self) -> None:
        """When tools are registered but the model returns no tool calls, the
        tool schemas are still advertised in the model request (proving it went
        through the tool-loop path), yet only one call is made."""
        responses = [
            ModelResponse(content="no tools needed", tool_calls=[]),
        ]

        provider = ScriptedProvider(responses)
        tool = RecordingTool("helper")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[tool],
        )
        built = agent.build()

        result = await built.arun("simple question")

        # One model call
        assert len(provider.requests) == 1
        # The request included tool schemas (proving tool-loop path was taken)
        request = provider.requests[0]
        assert request.tools is not None
        assert len(request.tools) > 0
        # Output is the direct model response
        assert result.output.text() == "no tools needed"
