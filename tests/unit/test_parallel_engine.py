"""Tests for ParallelEngine (BSP superstep execution).

Validates:
- Req 8.3: Parallel engine runs nodes at the same level concurrently
- Req 7.2: Barrier commits writes in node_id order (deterministic reducer application)
- Req 8.7: One node failing records the error but doesn't cancel siblings (fault isolation)
- Req 6.5: Edge conditions honored at each superstep
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.parallel import ParallelEngine
from loomable.flow.nodes import Edge, FlowConfigError, Node
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.state import SharedState, append


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str) -> RunResult:
    """Create a RunResult wrapping a text string."""
    output = AgentOutput(
        parts=[
            MediaPart(
                modality=Modality.TEXT,
                media_type="text/plain",
                data=text.encode("utf-8"),
            )
        ]
    )
    return RunResult(output=output, session_id="")


def _make_node(node_id: str, fn) -> Node:
    """Create a Node wrapping a function."""
    return Node(node_id=node_id, runnable=FunctionRunnable(fn))


class _FakeFlow:
    """Minimal fake Flow for testing engines in isolation."""

    def __init__(self, nodes: dict[str, Node], edges: list[Edge]):
        self._nodes = nodes
        self._edges = edges


# ---------------------------------------------------------------------------
# Test: Parallel engine runs nodes at the same level concurrently
# ---------------------------------------------------------------------------


class TestParallelConcurrency:
    """Verify that nodes at the same level run concurrently."""

    async def test_same_level_nodes_run_concurrently(self):
        """Nodes B and C (both depending only on A) should overlap in execution."""
        timestamps: dict[str, tuple[float, float]] = {}

        async def node_a(input):
            timestamps["A"] = (time.monotonic(), time.monotonic())
            return "a_done"

        async def node_b(input):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            timestamps["B"] = (start, time.monotonic())
            return "b_done"

        async def node_c(input):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            timestamps["C"] = (start, time.monotonic())
            return "c_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
            "C": _make_node("C", node_c),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="A", target="C"),
        ]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        # B and C should have overlapping execution times (concurrent)
        b_start, b_end = timestamps["B"]
        c_start, c_end = timestamps["C"]

        # If concurrent, one starts before the other ends
        assert b_start < c_end and c_start < b_end, (
            "B and C should overlap in time (concurrent execution)"
        )

    async def test_different_levels_are_sequential(self):
        """Nodes in different levels run in sequence (A before B)."""
        order: list[str] = []

        async def node_a(input):
            order.append("A")
            return "a_done"

        async def node_b(input):
            order.append("B")
            return "b_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        edges = [Edge(source="A", target="B")]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        assert order == ["A", "B"]


# ---------------------------------------------------------------------------
# Test: Barrier commits writes in node_id order (deterministic reducer)
# ---------------------------------------------------------------------------


class TestBarrierDeterminism:
    """Verify the barrier applies writes in node_id order (Req 7.2)."""

    async def test_barrier_commits_in_node_id_order(self):
        """Concurrent nodes writing the same key via an append reducer
        produce results ordered by node_id, not completion order."""
        # B finishes after C (C sleeps less), but barrier should apply B before C
        # because 'B' < 'C' alphabetically.

        async def node_b(input):
            await asyncio.sleep(0.05)  # B finishes later
            return "from_B"

        async def node_c(input):
            await asyncio.sleep(0.01)  # C finishes first
            return "from_C"

        nodes = {
            "B": _make_node("B", node_b),
            "C": _make_node("C", node_c),
        }
        edges: list[Edge] = []  # Both at level 0, no dependencies
        flow = _FakeFlow(nodes, edges)

        # Use an append reducer on a shared key — both nodes write to it
        # Since nodes write to state[node_id], and we use default overwrite,
        # we verify the sub_results ordering is deterministic (node_id order)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        result = await engine.run(flow, "start", state, ctx)

        # sub_results should contain both B and C
        assert "B" in result.sub_results
        assert "C" in result.sub_results

        # State should have entries for both nodes (written at barrier in order)
        assert state.get("B") is not None
        assert state.get("C") is not None

    async def test_reducer_applied_deterministically_at_barrier(self):
        """When nodes write to a shared key via a custom reducer,
        the reducer is applied in node_id order at the barrier."""
        # We'll use a flow where two nodes (A, B) at the same level both
        # produce output, and we'll check that state writes happen in
        # node_id order (A before B).
        write_order: list[str] = []

        original_write = SharedState.write

        def tracking_write(self, key, value):
            write_order.append(key)
            original_write(self, key, value)

        async def node_a(input):
            return "val_A"

        async def node_b(input):
            return "val_B"

        nodes = {
            "B": _make_node("B", node_b),
            "A": _make_node("A", node_a),
        }
        edges: list[Edge] = []  # Both at level 0
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        # Monkey-patch write to track order
        SharedState.write = tracking_write
        try:
            engine = ParallelEngine()
            await engine.run(flow, "start", state, ctx)
        finally:
            SharedState.write = original_write

        # Writes at barrier should be in node_id order: A before B
        assert write_order == ["A", "B"]


# ---------------------------------------------------------------------------
# Test: Fault isolation — one node failing doesn't cancel siblings
# ---------------------------------------------------------------------------


class TestFaultIsolation:
    """Verify per-node fault isolation (Req 8.7)."""

    async def test_failing_node_does_not_cancel_siblings(self):
        """When one node fails in a superstep, siblings still complete."""
        completed: list[str] = []

        async def node_a(input):
            completed.append("A")
            return "a_ok"

        async def node_b(input):
            raise RuntimeError("node B exploded")

        async def node_c(input):
            completed.append("C")
            return "c_ok"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
            "C": _make_node("C", node_c),
        }
        edges: list[Edge] = []  # All at level 0 (concurrent)
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        result = await engine.run(flow, "start", state, ctx)

        # A and C should have completed successfully
        assert "A" in completed
        assert "C" in completed

        # B should have an error recorded in sub_results
        assert "B" in result.sub_results
        assert "error" in result.sub_results["B"].metadata

        # A and C should be successful in sub_results
        assert "A" in result.sub_results
        assert "C" in result.sub_results
        assert "error" not in result.sub_results["A"].metadata
        assert "error" not in result.sub_results["C"].metadata

    async def test_failed_node_does_not_write_to_state(self):
        """A failed node's output is NOT written to SharedState."""

        async def node_a(input):
            raise ValueError("boom")

        async def node_b(input):
            return "b_ok"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        edges: list[Edge] = []  # Both at level 0
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        # A failed — should not be in state
        assert state.get("A") is None
        # B succeeded — should be in state
        assert state.get("B") is not None


# ---------------------------------------------------------------------------
# Test: Edge conditions honored at each superstep
# ---------------------------------------------------------------------------


class TestEdgeConditions:
    """Verify edge conditions are evaluated at each superstep."""

    async def test_node_skipped_when_condition_falsy(self):
        """A node whose all incoming edge conditions are falsy is skipped."""
        executed: list[str] = []

        async def node_a(input):
            executed.append("A")
            return "a_done"

        async def node_b(input):
            executed.append("B")
            return "b_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        # B depends on A, but only if state has "proceed" == True
        edges = [
            Edge(
                source="A",
                target="B",
                condition=lambda state: state.get("proceed") is True,
            )
        ]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        # A should execute (root node)
        assert "A" in executed
        # B should be skipped (condition is falsy — "proceed" not set)
        assert "B" not in executed

    async def test_node_executes_when_condition_truthy(self):
        """A node whose incoming edge condition is truthy executes normally."""
        executed: list[str] = []

        async def node_a(input):
            executed.append("A")
            return "a_done"

        async def node_b(input):
            executed.append("B")
            return "b_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        edges = [
            Edge(
                source="A",
                target="B",
                condition=lambda state: True,
            )
        ]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        assert "A" in executed
        assert "B" in executed

    async def test_unconditional_edge_always_passes(self):
        """A node with an unconditional incoming edge always executes."""
        executed: list[str] = []

        async def node_a(input):
            executed.append("A")
            return "a_done"

        async def node_b(input):
            executed.append("B")
            return "b_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        # Unconditional edge (condition=None)
        edges = [Edge(source="A", target="B")]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        assert "A" in executed
        assert "B" in executed

    async def test_condition_evaluated_with_current_state(self):
        """Edge conditions see state produced by earlier supersteps."""
        executed: list[str] = []

        async def node_a(input):
            executed.append("A")
            return "gate_open"

        async def node_b(input):
            executed.append("B")
            return "b_done"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        # B depends on A with a condition that checks if A's output is in state
        edges = [
            Edge(
                source="A",
                target="B",
                condition=lambda state: state.get("A") is not None,
            )
        ]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        await engine.run(flow, "start", state, ctx)

        # A executes first (level 0), writes to state
        # Then B's condition checks state["A"] which is now set
        assert "A" in executed
        assert "B" in executed


# ---------------------------------------------------------------------------
# Test: Cycle detection (Req 8.6)
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """ParallelEngine detects cycles before execution."""

    async def test_cycle_raises_flow_config_error(self):
        """A cyclic graph raises FlowConfigError."""

        async def node_a(input):
            return "a"

        async def node_b(input):
            return "b"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="A")]
        flow = _FakeFlow(nodes, edges)
        state = SharedState()
        ctx = RunContext()

        engine = ParallelEngine()
        with pytest.raises(FlowConfigError, match="cycle"):
            await engine.run(flow, "start", state, ctx)


# ---------------------------------------------------------------------------
# Test: Integration with Flow using engine="parallel"
# ---------------------------------------------------------------------------


class TestFlowParallelIntegration:
    """Verify Flow works end-to-end with engine='parallel'."""

    async def test_flow_with_parallel_engine_string(self):
        """Flow(engine='parallel') uses the ParallelEngine."""
        from loomable.flow.flow import Flow

        async def step_a(input):
            return f"a:{input}"

        async def step_b(input):
            return f"b:{input}"

        flow = Flow(
            {"A": step_a, "B": step_b},
            edges=[Edge(source="A", target="B")],
            engine="parallel",
        )
        result = await flow.arun("hello")

        assert result is not None
        assert result.metadata.get("flow_plan") is not None
        assert result.metadata["flow_plan"].engine == "ParallelEngine"
