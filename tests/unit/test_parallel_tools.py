"""Unit tests for high-level parallel tool calling (task 8.2).

Verify that ``BuiltAgent.dispatch_tools`` surfaces the kernel ToolRuntime's
concurrent, matched, fault-isolated dispatch through the high-level API (Req 12):

- All calls return exactly one outcome (Req 12.1).
- Each outcome's ``call_id`` matches its originating call (Req 12.2).
- A single failing tool is isolated: it yields an error outcome while siblings
  still return successful results (Req 12.3).
- Dispatch reuses the kernel ToolRuntime unchanged and runs concurrently (Req 12.4).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from loomable.agent import Agent
from loomable.kernel.contracts import Tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation (satisfies the structural protocol)."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


class _EchoTool(Tool):
    """A tool that echoes its ``value`` argument back as the result content."""

    name = "echo"
    description = "Echoes the provided value."

    async def invoke(self, args: dict) -> ToolResult:
        return ToolResult(content=args.get("value"))


class _FailingTool(Tool):
    """A tool that always raises, to exercise fault isolation."""

    name = "boom"
    description = "Always fails."

    async def invoke(self, args: dict) -> ToolResult:
        raise RuntimeError("tool exploded")


class _SlowTool(Tool):
    """A tool that sleeps for ``delay`` seconds before returning, to show concurrency."""

    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.description = f"Sleeps {delay}s then echoes."
        self._delay = delay

    async def invoke(self, args: dict) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(content=self.name)


def _build_agent(*tools: Tool):
    """Build a BuiltAgent whose ToolRuntime is populated with the given tools."""
    return Agent(model=_FakeProvider(), tools=list(tools)).build()


# ---------------------------------------------------------------------------
# Matching + completeness (Req 12.1, 12.2)
# ---------------------------------------------------------------------------


class TestDispatchToolsMatching:
    async def test_all_calls_return_matched_outcomes(self):
        built = _build_agent(_EchoTool())
        calls = [
            ToolCall(id="c1", tool_name="echo", args={"value": "a"}),
            ToolCall(id="c2", tool_name="echo", args={"value": "b"}),
            ToolCall(id="c3", tool_name="echo", args={"value": "c"}),
        ]

        outcomes = await built.dispatch_tools(calls)

        # One outcome per call (Req 12.1).
        assert len(outcomes) == len(calls)
        # Each outcome matched to its originating call id, order preserved (Req 12.2).
        assert [o.call_id for o in outcomes] == ["c1", "c2", "c3"]
        # Results carry the echoed argument.
        assert [o.result.content for o in outcomes] == ["a", "b", "c"]
        assert all(o.error is None for o in outcomes)

    async def test_empty_calls_returns_empty(self):
        built = _build_agent(_EchoTool())
        assert await built.dispatch_tools([]) == []


# ---------------------------------------------------------------------------
# Fault isolation (Req 12.3)
# ---------------------------------------------------------------------------


class TestDispatchToolsFaultIsolation:
    async def test_one_failure_does_not_cancel_siblings(self):
        built = _build_agent(_EchoTool(), _FailingTool())
        calls = [
            ToolCall(id="ok-1", tool_name="echo", args={"value": "x"}),
            ToolCall(id="bad", tool_name="boom", args={}),
            ToolCall(id="ok-2", tool_name="echo", args={"value": "y"}),
        ]

        outcomes = await built.dispatch_tools(calls)
        by_id = {o.call_id: o for o in outcomes}

        # Every call still produced an outcome.
        assert set(by_id) == {"ok-1", "bad", "ok-2"}
        # Siblings succeeded.
        assert by_id["ok-1"].error is None
        assert by_id["ok-1"].result.content == "x"
        assert by_id["ok-2"].error is None
        assert by_id["ok-2"].result.content == "y"
        # The failing call is isolated as an error outcome.
        assert by_id["bad"].result is None
        assert by_id["bad"].error is not None
        assert "tool exploded" in by_id["bad"].error.message

    async def test_unknown_tool_is_isolated_error(self):
        built = _build_agent(_EchoTool())
        calls = [
            ToolCall(id="ok", tool_name="echo", args={"value": "z"}),
            ToolCall(id="missing", tool_name="does_not_exist", args={}),
        ]

        outcomes = await built.dispatch_tools(calls)
        by_id = {o.call_id: o for o in outcomes}

        assert by_id["ok"].result.content == "z"
        assert by_id["missing"].error is not None


# ---------------------------------------------------------------------------
# Concurrency (Req 12.1, 12.4)
# ---------------------------------------------------------------------------


class TestDispatchToolsConcurrency:
    async def test_independent_calls_run_concurrently(self):
        delay = 0.1
        built = _build_agent(
            _SlowTool("slow_a", delay),
            _SlowTool("slow_b", delay),
            _SlowTool("slow_c", delay),
        )
        calls = [
            ToolCall(id="a", tool_name="slow_a"),
            ToolCall(id="b", tool_name="slow_b"),
            ToolCall(id="c", tool_name="slow_c"),
        ]

        start = time.perf_counter()
        outcomes = await built.dispatch_tools(calls)
        elapsed = time.perf_counter() - start

        assert len(outcomes) == 3
        assert all(o.error is None for o in outcomes)
        # Concurrent execution: wall-clock is far less than the serial sum (3 * delay).
        assert elapsed < delay * len(calls)
