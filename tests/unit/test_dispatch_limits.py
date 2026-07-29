"""Unit tests for per-tool timeout and concurrency cap in gated dispatch (Req 2.1–2.4).

Covers:
- A tool that exceeds tool_timeout produces a ToolOutcome with ToolError naming the tool.
- Sibling calls in the same batch complete even when one times out.
- Concurrency cap limits simultaneous in-flight tool invocations.
- When neither timeout nor concurrency is configured, dispatch goes directly to the
  kernel runtime (unchanged behavior).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from loomable.agent import Agent, GatedDispatchResult, ModelSpec
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolOutcome,
    ToolResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider (satisfies the structural protocol)."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


class SlowTool(Tool):
    """A tool that sleeps for a configurable duration."""

    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.description = f"Sleeps for {delay}s."
        self.delay = delay

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(self.delay)
        return ToolResult(content={"tool": self.name, "slept": self.delay})


class ConcurrencyTracker(Tool):
    """A tool that tracks max concurrent invocations."""

    def __init__(self, name: str, delay: float = 0.05) -> None:
        self.name = name
        self.description = f"Tracks concurrency ({name})."
        self.delay = delay
        self.current = 0
        self.peak = 0

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.current += 1
        if self.current > self.peak:
            self.peak = self.current
        await asyncio.sleep(self.delay)
        self.current -= 1
        return ToolResult(content={"peak": self.peak})


class FastTool(Tool):
    """A tool that returns immediately."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "Returns immediately."

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content={"tool": self.name})


def _agent_with_tools(tools: list[Tool], **kwargs: Any):
    """Build a BuiltAgent over the given tools using a fake provider."""
    return Agent(
        model=ModelSpec(provider="fake", provider_impl=_FakeProvider()),
        capabilities=ModelCapabilities(),
        tools=tools,
        **kwargs,
    ).build()


# ---------------------------------------------------------------------------
# Per-tool timeout (Req 2.1)
# ---------------------------------------------------------------------------


class TestPerToolTimeout:
    async def test_timeout_produces_tool_error_naming_tool(self) -> None:
        """A tool exceeding tool_timeout produces a ToolOutcome with ToolError (Req 2.1)."""
        slow = SlowTool("slow_tool", delay=1.0)
        built = _agent_with_tools([slow])
        built.tool_timeout = 0.05  # 50ms timeout

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="slow_tool", args={})]
        )

        assert len(result.outcomes) == 1
        outcome = result.outcomes[0]
        assert outcome.call_id == "c1"
        assert outcome.error is not None
        assert outcome.result is None
        assert "slow_tool" in outcome.error.message
        assert "timed out" in outcome.error.message

    async def test_sibling_calls_complete_when_one_times_out(self) -> None:
        """Sibling calls still complete even when one tool times out (Req 2.2)."""
        slow = SlowTool("slow_tool", delay=1.0)
        fast = FastTool("fast_tool")
        built = _agent_with_tools([slow, fast])
        built.tool_timeout = 0.05  # 50ms timeout

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="c1", tool_name="slow_tool", args={}),
                ToolCall(id="c2", tool_name="fast_tool", args={}),
            ]
        )

        assert len(result.outcomes) == 2
        # slow_tool timed out
        slow_outcome = next(o for o in result.outcomes if o.call_id == "c1")
        assert slow_outcome.error is not None
        assert "slow_tool" in slow_outcome.error.message
        assert "timed out" in slow_outcome.error.message
        # fast_tool succeeded
        fast_outcome = next(o for o in result.outcomes if o.call_id == "c2")
        assert fast_outcome.result is not None
        assert fast_outcome.error is None

    async def test_no_timeout_when_tool_completes_in_time(self) -> None:
        """A tool that completes within the timeout produces a normal result."""
        fast = FastTool("fast_tool")
        built = _agent_with_tools([fast])
        built.tool_timeout = 5.0  # generous timeout

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="fast_tool", args={})]
        )

        assert len(result.outcomes) == 1
        outcome = result.outcomes[0]
        assert outcome.result is not None
        assert outcome.error is None

    async def test_timeout_error_message_includes_duration(self) -> None:
        """The timeout error message includes the configured timeout value."""
        slow = SlowTool("my_tool", delay=1.0)
        built = _agent_with_tools([slow])
        built.tool_timeout = 0.02

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="my_tool", args={})]
        )

        outcome = result.outcomes[0]
        assert "0.02s" in outcome.error.message


# ---------------------------------------------------------------------------
# Concurrency cap (Req 2.3)
# ---------------------------------------------------------------------------


class TestConcurrencyCap:
    async def test_concurrency_cap_limits_parallel_calls(self) -> None:
        """The concurrency cap limits simultaneous in-flight invocations (Req 2.3)."""
        tracker = ConcurrencyTracker("tracker", delay=0.05)
        built = _agent_with_tools([tracker])
        built.tool_concurrency = 2  # max 2 at a time

        # Launch 4 calls; with cap=2, peak should be ≤ 2.
        calls = [
            ToolCall(id=f"c{i}", tool_name="tracker", args={"i": i})
            for i in range(4)
        ]
        result = await built.dispatch_tools_gated(calls)

        assert len(result.outcomes) == 4
        # All calls should succeed
        for outcome in result.outcomes:
            assert outcome.result is not None
        # Peak concurrent should be at most 2
        assert tracker.peak <= 2

    async def test_no_cap_allows_full_parallelism(self) -> None:
        """Without a concurrency cap, all calls run in parallel."""
        tracker = ConcurrencyTracker("tracker", delay=0.05)
        built = _agent_with_tools([tracker])
        # No concurrency cap, but set timeout so _dispatch_with_limits is used
        built.tool_timeout = 5.0

        calls = [
            ToolCall(id=f"c{i}", tool_name="tracker", args={"i": i})
            for i in range(4)
        ]
        result = await built.dispatch_tools_gated(calls)

        assert len(result.outcomes) == 4
        for outcome in result.outcomes:
            assert outcome.result is not None
        # Without a cap, peak should reach the full batch size
        assert tracker.peak >= 3  # Allow some scheduling slack


# ---------------------------------------------------------------------------
# Combined timeout + concurrency (Req 2.1–2.3)
# ---------------------------------------------------------------------------


class TestCombinedLimits:
    async def test_timeout_and_concurrency_together(self) -> None:
        """Both timeout and concurrency cap work together."""
        slow = SlowTool("slow", delay=1.0)
        fast = FastTool("fast")
        built = _agent_with_tools([slow, fast])
        built.tool_timeout = 0.05
        built.tool_concurrency = 1  # serial execution

        result = await built.dispatch_tools_gated(
            [
                ToolCall(id="c1", tool_name="slow", args={}),
                ToolCall(id="c2", tool_name="fast", args={}),
            ]
        )

        assert len(result.outcomes) == 2
        slow_outcome = next(o for o in result.outcomes if o.call_id == "c1")
        fast_outcome = next(o for o in result.outcomes if o.call_id == "c2")
        # Slow timed out
        assert slow_outcome.error is not None
        assert "timed out" in slow_outcome.error.message
        # Fast succeeded
        assert fast_outcome.result is not None


# ---------------------------------------------------------------------------
# No limits configured — direct dispatch (unchanged behavior)
# ---------------------------------------------------------------------------


class TestNoLimitsConfigured:
    async def test_no_limits_dispatches_directly(self) -> None:
        """Without timeout or concurrency, dispatch goes directly to the kernel."""
        fast = FastTool("fast")
        built = _agent_with_tools([fast])
        # Neither tool_timeout nor tool_concurrency set (both None)

        result = await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="fast", args={})]
        )

        assert len(result.outcomes) == 1
        assert result.outcomes[0].result is not None

    async def test_tool_not_retried_on_timeout(self) -> None:
        """A timed-out tool is called exactly once — never blind-retried (Req 2.4)."""
        call_count = 0

        class CountingTool(Tool):
            name = "counter"
            description = "Counts invocations."

            async def invoke(self, args: dict[str, Any]) -> ToolResult:
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(1.0)
                return ToolResult(content="done")

        built = _agent_with_tools([CountingTool()])
        built.tool_timeout = 0.02

        await built.dispatch_tools_gated(
            [ToolCall(id="c1", tool_name="counter", args={})]
        )

        assert call_count == 1  # exactly one attempt, no retry
