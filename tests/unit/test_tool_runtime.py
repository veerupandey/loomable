"""Unit tests for ToolRuntime concurrent dispatch with isolation.

Tests verify:
- Concurrent dispatch of multiple tool calls
- Each ToolOutcome carries its originating tool_call_id
- One failure does NOT cancel siblings (isolation)
- Unknown tool names produce an error outcome
- Empty call list returns empty outcomes
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from loomable.kernel.contracts import Tool
from loomable.kernel.models import ToolCall, ToolOutcome, ToolResult
from loomable.kernel.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Test helpers / fake tools
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """A tool that echoes back its arguments."""

    name = "echo"
    description = "Echoes the input arguments."

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=args)


class FailingTool(Tool):
    """A tool that always raises an exception."""

    name = "failing"
    description = "Always fails."

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("Tool execution failed deliberately")


class SlowTool(Tool):
    """A tool that sleeps for a configurable duration before returning."""

    name = "slow"
    description = "Sleeps then returns."

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(content={"delayed": self._delay, **args})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolRuntimeDispatch:
    """Tests for ToolRuntime.dispatch()."""

    @pytest.fixture
    def runtime(self) -> ToolRuntime:
        """Runtime with echo, failing, and slow tools."""
        tools: dict[str, Tool] = {
            "echo": EchoTool(),
            "failing": FailingTool(),
            "slow": SlowTool(delay=0.05),
        }
        return ToolRuntime(tools)

    async def test_empty_calls_returns_empty(self, runtime: ToolRuntime) -> None:
        """Dispatching an empty list returns an empty list."""
        outcomes = await runtime.dispatch([])
        assert outcomes == []

    async def test_single_successful_call(self, runtime: ToolRuntime) -> None:
        """A single successful tool call returns a ToolOutcome with result."""
        calls = [ToolCall(id="call-1", tool_name="echo", args={"msg": "hello"})]
        outcomes = await runtime.dispatch(calls)

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.call_id == "call-1"
        assert outcome.result is not None
        assert outcome.error is None
        assert outcome.result.content == {"msg": "hello"}

    async def test_single_failing_call(self, runtime: ToolRuntime) -> None:
        """A single failing tool call returns a ToolOutcome with error."""
        calls = [ToolCall(id="call-2", tool_name="failing", args={})]
        outcomes = await runtime.dispatch(calls)

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.call_id == "call-2"
        assert outcome.error is not None
        assert outcome.result is None
        assert "RuntimeError" in outcome.error.message

    async def test_unknown_tool_returns_error(self, runtime: ToolRuntime) -> None:
        """An unknown tool name returns a ToolOutcome with an error."""
        calls = [ToolCall(id="call-3", tool_name="nonexistent", args={})]
        outcomes = await runtime.dispatch(calls)

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.call_id == "call-3"
        assert outcome.error is not None
        assert "not found" in outcome.error.message.lower()

    async def test_failure_does_not_cancel_siblings(self, runtime: ToolRuntime) -> None:
        """One failing call does NOT cancel other concurrent calls."""
        calls = [
            ToolCall(id="ok-1", tool_name="echo", args={"x": 1}),
            ToolCall(id="fail-1", tool_name="failing", args={}),
            ToolCall(id="ok-2", tool_name="echo", args={"x": 2}),
        ]
        outcomes = await runtime.dispatch(calls)

        assert len(outcomes) == 3

        # First call succeeds
        assert outcomes[0].call_id == "ok-1"
        assert outcomes[0].result is not None
        assert outcomes[0].result.content == {"x": 1}

        # Second call fails
        assert outcomes[1].call_id == "fail-1"
        assert outcomes[1].error is not None

        # Third call succeeds despite sibling failure
        assert outcomes[2].call_id == "ok-2"
        assert outcomes[2].result is not None
        assert outcomes[2].result.content == {"x": 2}

    async def test_call_id_matches_originating_call(self, runtime: ToolRuntime) -> None:
        """Each outcome's call_id matches its originating ToolCall.id."""
        calls = [
            ToolCall(id="alpha", tool_name="echo", args={"v": "a"}),
            ToolCall(id="beta", tool_name="echo", args={"v": "b"}),
            ToolCall(id="gamma", tool_name="echo", args={"v": "c"}),
        ]
        outcomes = await runtime.dispatch(calls)

        assert len(outcomes) == 3
        for call, outcome in zip(calls, outcomes):
            assert outcome.call_id == call.id

    async def test_concurrent_execution_overlap(self) -> None:
        """Calls execute concurrently (total time < sum of individual delays)."""
        delay = 0.1
        num_calls = 3
        tools: dict[str, Tool] = {"slow": SlowTool(delay=delay)}
        runtime = ToolRuntime(tools)

        calls = [
            ToolCall(id=f"s-{i}", tool_name="slow", args={"i": i})
            for i in range(num_calls)
        ]

        start = time.monotonic()
        outcomes = await runtime.dispatch(calls)
        elapsed = time.monotonic() - start

        # All should succeed
        assert len(outcomes) == num_calls
        for outcome in outcomes:
            assert outcome.result is not None

        # Total time should be much less than serial execution
        serial_time = delay * num_calls
        assert elapsed < serial_time * 0.8  # At least 20% faster than serial

    async def test_outcomes_preserve_order(self, runtime: ToolRuntime) -> None:
        """Outcomes are returned in the same order as input calls."""
        calls = [
            ToolCall(id="first", tool_name="slow", args={}),
            ToolCall(id="second", tool_name="echo", args={}),
            ToolCall(id="third", tool_name="echo", args={}),
        ]
        outcomes = await runtime.dispatch(calls)

        assert [o.call_id for o in outcomes] == ["first", "second", "third"]
