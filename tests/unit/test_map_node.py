"""Tests for MapNode (Task 9.1).

Validates Req 11.1, 11.2, 11.3:
- MapNode fans out over a list from state and collects results
- Concurrency cap limits simultaneous executions
- Per-item failure isolation (one item failing doesn't cancel others)
- MapNode satisfies Runnable protocol
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow import MapNode
from loomable.flow.runnable import Runnable
from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EchoRunnable:
    """A simple Runnable that echoes its input as a RunResult."""

    async def arun(self, input: Any, *, context: RunContext | None = None) -> RunResult:
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=f"processed:{input}".encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")


class FailingRunnable:
    """A Runnable that always raises an error."""

    async def arun(self, input: Any, *, context: RunContext | None = None) -> RunResult:
        raise ValueError(f"failed on: {input}")


class ConditionalFailRunnable:
    """A Runnable that fails on specific items (those containing 'fail')."""

    async def arun(self, input: Any, *, context: RunContext | None = None) -> RunResult:
        if "fail" in str(input):
            raise ValueError(f"intentional failure on: {input}")
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=f"ok:{input}".encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")


class ConcurrencyTracker:
    """A Runnable that tracks how many concurrent executions happen."""

    def __init__(self) -> None:
        self.max_concurrent = 0
        self._current = 0
        self._lock = asyncio.Lock()

    async def arun(self, input: Any, *, context: RunContext | None = None) -> RunResult:
        async with self._lock:
            self._current += 1
            if self._current > self.max_concurrent:
                self.max_concurrent = self._current

        # Simulate some async work
        await asyncio.sleep(0.05)

        async with self._lock:
            self._current -= 1

        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=f"done:{input}".encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")


def _make_context_with_items(items: list[Any], key: str = "items") -> RunContext:
    """Create a RunContext with a SharedState containing the given items list."""
    state = SharedState()
    state.write(key, items)
    return RunContext(shared_state=state)


# ---------------------------------------------------------------------------
# Tests: MapNode satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestMapNodeProtocol:
    def test_map_node_satisfies_runnable_protocol(self):
        """MapNode must satisfy the Runnable protocol."""
        node = MapNode(EchoRunnable(), over="items")
        assert isinstance(node, Runnable)

    def test_map_node_has_arun(self):
        """MapNode must have an async arun method."""
        import inspect

        node = MapNode(EchoRunnable(), over="items")
        assert hasattr(node, "arun")
        assert inspect.iscoroutinefunction(node.arun)


# ---------------------------------------------------------------------------
# Tests: MapNode fans out over a list and collects results
# ---------------------------------------------------------------------------


class TestMapNodeFanOut:
    @pytest.mark.asyncio
    async def test_fans_out_over_list_from_state(self):
        """Req 11.1: MapNode reads a list from state and runs body per item."""
        node = MapNode(EchoRunnable(), over="items")
        ctx = _make_context_with_items(["a", "b", "c"])

        result = await node.arun("ignored_input", context=ctx)

        assert result.metadata["map_total"] == 3
        assert result.metadata["map_succeeded"] == 3
        assert result.metadata["map_failed"] == 0

    @pytest.mark.asyncio
    async def test_results_fanned_into_collection(self):
        """Req 11.2: Results are fanned back into a single collection."""
        node = MapNode(EchoRunnable(), over="items")
        ctx = _make_context_with_items(["x", "y"])

        result = await node.arun("input", context=ctx)

        map_results = result.metadata["map_results"]
        assert len(map_results) == 2
        # All should be successful
        assert all(r["success"] for r in map_results)
        # Each result should contain a RunResult object
        for r in map_results:
            assert isinstance(r["result"], RunResult)

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_result(self):
        """MapNode with an empty list returns an empty result."""
        node = MapNode(EchoRunnable(), over="items")
        ctx = _make_context_with_items([])

        result = await node.arun("input", context=ctx)

        assert result.metadata["map_results"] == []
        assert result.metadata["map_errors"] == []
        assert result.metadata["map_total"] == 0
        assert result.metadata["map_succeeded"] == 0
        assert result.output.text() == ""
        assert ctx.shared_state.get("map") == []

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        """MapNode with a missing state key raises FlowConfigError."""
        from loomable.flow.nodes import FlowConfigError

        node = MapNode(EchoRunnable(), over="nonexistent_key")
        state = SharedState()
        ctx = RunContext(shared_state=state)

        with pytest.raises(FlowConfigError, match="nonexistent_key"):
            await node.arun("input", context=ctx)

    @pytest.mark.asyncio
    async def test_no_context_raises(self):
        """MapNode without shared state raises FlowConfigError."""
        from loomable.flow.nodes import FlowConfigError

        node = MapNode(EchoRunnable(), over="items")

        with pytest.raises(FlowConfigError, match="shared_state"):
            await node.arun("input", context=None)

    @pytest.mark.asyncio
    async def test_result_is_run_result(self):
        """MapNode.arun returns a RunResult."""
        node = MapNode(EchoRunnable(), over="items")
        ctx = _make_context_with_items(["a"])

        result = await node.arun("input", context=ctx)

        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Tests: Concurrency cap limits simultaneous executions
# ---------------------------------------------------------------------------


class TestMapNodeConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_cap_limits_parallel_execution(self):
        """Concurrency cap limits the number of simultaneous executions."""
        tracker = ConcurrencyTracker()
        node = MapNode(tracker, over="items", concurrency=2)
        ctx = _make_context_with_items(["a", "b", "c", "d", "e"])

        await node.arun("input", context=ctx)

        # The max concurrent should not exceed the cap
        assert tracker.max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_no_concurrency_cap_allows_all_parallel(self):
        """Without a concurrency cap, all items run in parallel."""
        tracker = ConcurrencyTracker()
        node = MapNode(tracker, over="items", concurrency=None)
        ctx = _make_context_with_items(["a", "b", "c", "d", "e"])

        await node.arun("input", context=ctx)

        # Without a cap, all 5 should run concurrently
        assert tracker.max_concurrent >= 3  # at least most run concurrently

    @pytest.mark.asyncio
    async def test_concurrency_cap_of_one_runs_sequentially(self):
        """Concurrency=1 effectively runs items sequentially."""
        tracker = ConcurrencyTracker()
        node = MapNode(tracker, over="items", concurrency=1)
        ctx = _make_context_with_items(["a", "b", "c"])

        await node.arun("input", context=ctx)

        assert tracker.max_concurrent == 1


# ---------------------------------------------------------------------------
# Tests: Per-item failure isolation
# ---------------------------------------------------------------------------


class TestMapNodeFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_failure_does_not_cancel_others(self):
        """Req 11.3: One item failing doesn't cancel others."""
        node = MapNode(ConditionalFailRunnable(), over="items")
        ctx = _make_context_with_items(["ok1", "fail_item", "ok2"])

        result = await node.arun("input", context=ctx)

        # 2 should succeed, 1 should fail
        assert result.metadata["map_succeeded"] == 2
        assert result.metadata["map_failed"] == 1
        assert result.metadata["map_total"] == 3

    @pytest.mark.asyncio
    async def test_failure_recorded_with_error_metadata(self):
        """Failed items are recorded with error metadata."""
        node = MapNode(ConditionalFailRunnable(), over="items")
        ctx = _make_context_with_items(["ok1", "fail_item"])

        result = await node.arun("input", context=ctx)

        errors = result.metadata["map_errors"]
        assert len(errors) == 1
        assert errors[0]["index"] == 1
        assert "fail_item" in errors[0]["cause"]

    @pytest.mark.asyncio
    async def test_all_failures_still_returns_result(self):
        """Even if all items fail, MapNode returns a result (not an exception)."""
        node = MapNode(FailingRunnable(), over="items")
        ctx = _make_context_with_items(["a", "b", "c"])

        result = await node.arun("input", context=ctx)

        assert result.metadata["map_failed"] == 3
        assert result.metadata["map_succeeded"] == 0
        assert len(result.metadata["map_errors"]) == 3

    @pytest.mark.asyncio
    async def test_mixed_results_preserved_in_order(self):
        """Results maintain order correspondence with input items."""
        node = MapNode(ConditionalFailRunnable(), over="items")
        ctx = _make_context_with_items(["ok1", "fail_x", "ok2", "fail_y", "ok3"])

        result = await node.arun("input", context=ctx)

        map_results = result.metadata["map_results"]
        assert len(map_results) == 5

        # Verify order: index 0=ok, 1=fail, 2=ok, 3=fail, 4=ok
        assert map_results[0]["success"] is True
        assert map_results[1]["success"] is False
        assert map_results[2]["success"] is True
        assert map_results[3]["success"] is False
        assert map_results[4]["success"] is True


# ---------------------------------------------------------------------------
# Tests: MapNode repr
# ---------------------------------------------------------------------------


class TestMapNodeRepr:
    def test_repr_without_concurrency(self):
        node = MapNode(EchoRunnable(), over="items")
        assert repr(node) == "MapNode(over='items')"

    def test_repr_with_concurrency(self):
        node = MapNode(EchoRunnable(), over="tasks", concurrency=3)
        assert repr(node) == "MapNode(over='tasks', concurrency=3)"
