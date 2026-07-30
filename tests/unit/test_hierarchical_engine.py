"""Tests for HierarchicalEngine (manager/worker delegation).

Validates:
- Req 8.4: Hierarchical engine runs a designated manager node which delegates
  to worker nodes and synthesizes their results.
- Req 8.7: Worker fault isolation — one worker failing doesn't cancel others.
- Integration with Flow using engine="hierarchical".
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.hierarchical import HierarchicalEngine
from loomable.flow.nodes import Edge, FlowConfigError, Node
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.state import SharedState


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


def _make_node(node_id: str, fn, *, manager: bool = False) -> Node:
    """Create a Node wrapping a function."""
    return Node(node_id=node_id, runnable=FunctionRunnable(fn), manager=manager)


class _FakeFlow:
    """Minimal fake Flow for testing engines in isolation."""

    def __init__(self, nodes: dict[str, Node], edges: list[Edge] | None = None):
        self._nodes = nodes
        self._edges = edges or []


# ---------------------------------------------------------------------------
# Test: Manager delegates to workers, sees their results, and synthesizes
# ---------------------------------------------------------------------------


class TestHierarchicalDelegation:
    """Verify manager delegates to workers and synthesizes results."""

    async def test_manager_sees_worker_results_via_state(self):
        """The manager node can access worker outputs through shared state."""
        async def worker_a(input):
            return f"research:{input}"

        async def worker_b(input):
            return f"analysis:{input}"

        async def synthesizer(input, *, context=None):
            # Manager reads worker results from shared state
            state = context.shared_state
            a_result = state.get("worker_a")
            b_result = state.get("worker_b")
            # Synthesize results
            return f"synthesis({a_result},{b_result})"

        nodes = {
            "worker_a": _make_node("worker_a", worker_a),
            "worker_b": _make_node("worker_b", worker_b),
            "synthesizer": _make_node("synthesizer", synthesizer, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()
        ctx.shared_state = state

        engine = HierarchicalEngine()
        result = await engine.run(flow, "topic", state, ctx)

        # The final output comes from the manager
        output_text = result.output.parts[0].data.decode("utf-8")
        assert "synthesis(" in output_text

        # sub_results contains all nodes
        assert "worker_a" in result.sub_results
        assert "worker_b" in result.sub_results
        assert "synthesizer" in result.sub_results

    async def test_workers_run_concurrently(self):
        """Worker nodes execute concurrently via SubagentManager."""
        import time

        timestamps: dict[str, tuple[float, float]] = {}

        async def worker_a(input):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            timestamps["A"] = (start, time.monotonic())
            return "a_done"

        async def worker_b(input):
            start = time.monotonic()
            await asyncio.sleep(0.05)
            timestamps["B"] = (start, time.monotonic())
            return "b_done"

        async def mgr(input):
            return "synthesized"

        nodes = {
            "A": _make_node("A", worker_a),
            "B": _make_node("B", worker_b),
            "mgr": _make_node("mgr", mgr, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        await engine.run(flow, "start", state, ctx)

        # Workers should overlap in time (concurrent)
        a_start, a_end = timestamps["A"]
        b_start, b_end = timestamps["B"]
        assert a_start < b_end and b_start < a_end, (
            "Workers A and B should overlap in time (concurrent execution)"
        )

    async def test_manager_output_is_final_result(self):
        """The final RunResult uses the manager's output."""
        async def worker(input):
            return "worker_output"

        async def mgr(input):
            return "manager_final_output"

        nodes = {
            "worker": _make_node("worker", worker),
            "mgr": _make_node("mgr", mgr, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        result = await engine.run(flow, "start", state, ctx)

        output_text = result.output.parts[0].data.decode("utf-8")
        assert output_text == "manager_final_output"


# ---------------------------------------------------------------------------
# Test: Missing manager node raises FlowConfigError
# ---------------------------------------------------------------------------


class TestMissingManager:
    """Verify FlowConfigError when no manager node is found."""

    async def test_no_manager_raises_config_error(self):
        """A flow with no manager=True node raises FlowConfigError."""
        async def node_a(input):
            return "a"

        async def node_b(input):
            return "b"

        nodes = {
            "A": _make_node("A", node_a),
            "B": _make_node("B", node_b),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        with pytest.raises(FlowConfigError, match="manager=True"):
            await engine.run(flow, "start", state, ctx)

    async def test_multiple_managers_raises_config_error(self):
        """A flow with more than one manager=True node raises FlowConfigError."""
        async def node_a(input):
            return "a"

        async def node_b(input):
            return "b"

        nodes = {
            "A": _make_node("A", node_a, manager=True),
            "B": _make_node("B", node_b, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        with pytest.raises(FlowConfigError, match="multiple manager"):
            await engine.run(flow, "start", state, ctx)


# ---------------------------------------------------------------------------
# Test: Worker fault isolation
# ---------------------------------------------------------------------------


class TestWorkerFaultIsolation:
    """Verify one worker failing doesn't cancel others (Req 8.7)."""

    async def test_failing_worker_does_not_cancel_others(self):
        """One worker raising does not prevent other workers from completing."""
        completed: list[str] = []

        async def worker_a(input):
            completed.append("A")
            return "a_ok"

        async def worker_b(input):
            raise RuntimeError("worker B exploded")

        async def worker_c(input):
            completed.append("C")
            return "c_ok"

        async def mgr(input, *, context=None):
            # Manager can still see successful workers
            state = context.shared_state
            return f"mgr_got_a={state.get('worker_a') is not None}"

        nodes = {
            "worker_a": _make_node("worker_a", worker_a),
            "worker_b": _make_node("worker_b", worker_b),
            "worker_c": _make_node("worker_c", worker_c),
            "mgr": _make_node("mgr", mgr, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()
        ctx.shared_state = state

        engine = HierarchicalEngine()
        result = await engine.run(flow, "start", state, ctx)

        # Workers A and C should have completed
        assert "A" in completed
        assert "C" in completed

        # Worker B should be recorded with error in sub_results
        assert "worker_b" in result.sub_results
        assert "error" in result.sub_results["worker_b"].metadata

        # Workers A and C should be successful in sub_results
        assert "worker_a" in result.sub_results
        assert "worker_c" in result.sub_results
        assert "error" not in result.sub_results["worker_a"].metadata
        assert "error" not in result.sub_results["worker_c"].metadata

    async def test_failed_worker_does_not_write_to_state(self):
        """A failed worker's output is NOT written to SharedState."""
        async def worker_a(input):
            raise ValueError("boom")

        async def worker_b(input):
            return "b_ok"

        async def mgr(input):
            return "done"

        nodes = {
            "worker_a": _make_node("worker_a", worker_a),
            "worker_b": _make_node("worker_b", worker_b),
            "mgr": _make_node("mgr", mgr, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        await engine.run(flow, "start", state, ctx)

        # worker_a failed — should not be in state
        assert state.get("worker_a") is None
        # worker_b succeeded — should be in state
        assert state.get("worker_b") is not None


# ---------------------------------------------------------------------------
# Test: Integration with Flow using engine="hierarchical"
# ---------------------------------------------------------------------------


class TestFlowHierarchicalIntegration:
    """Verify Flow works end-to-end with engine='hierarchical'."""

    async def test_flow_with_hierarchical_engine_string(self):
        """Flow(engine='hierarchical') uses the HierarchicalEngine."""
        from loomable.flow.flow import Flow
        from loomable.flow.runnable import FunctionRunnable

        async def researcher(input):
            return f"researched:{input}"

        async def writer(input):
            return f"written:{input}"

        async def editor(input, *, context=None):
            state = context.shared_state
            r = state.get("researcher")
            w = state.get("writer")
            return f"edited({r},{w})"

        # Build nodes dict — the editor is the manager
        nodes_dict: dict[str, Any] = {
            "researcher": researcher,
            "writer": writer,
        }

        # Need to create flow with Node objects that have manager flag
        # Use the dict interface but we need the manager flag on one node
        # For this, we need to construct manually via Node objects
        from loomable.flow.nodes import Node as FlowNode

        researcher_node = FlowNode(
            node_id="researcher", runnable=FunctionRunnable(researcher)
        )
        writer_node = FlowNode(
            node_id="writer", runnable=FunctionRunnable(writer)
        )
        editor_node = FlowNode(
            node_id="editor", runnable=FunctionRunnable(editor), manager=True
        )

        # Use a custom flow that bypasses Flow.__init__ dict handling
        # since Flow's dict constructor doesn't set manager flag
        flow = Flow.__new__(Flow)
        flow._engine = "hierarchical"
        flow._optimizer = False
        flow._memory = None
        flow._checkpointer = None
        flow._events = None
        flow._session_id = None
        flow._deps = None
        flow._reducers = None
        flow._nodes = {
            "researcher": researcher_node,
            "writer": writer_node,
            "editor": editor_node,
        }
        flow._edges = []

        result = await flow.arun("topic")

        assert result is not None
        assert result.metadata.get("flow_plan") is not None
        assert result.metadata["flow_plan"].engine == "HierarchicalEngine"

        # The final output should be from the editor (manager)
        output_text = result.output.parts[0].data.decode("utf-8")
        assert "edited(" in output_text

    async def test_flow_hierarchical_with_no_workers(self):
        """A hierarchical flow with only a manager node works (edge case)."""
        async def solo_manager(input):
            return f"solo:{input}"

        nodes = {
            "mgr": _make_node("mgr", solo_manager, manager=True),
        }
        flow = _FakeFlow(nodes)
        state = SharedState()
        ctx = RunContext()

        engine = HierarchicalEngine()
        result = await engine.run(flow, "hello", state, ctx)

        output_text = result.output.parts[0].data.decode("utf-8")
        assert output_text == "solo:hello"
        assert "mgr" in result.sub_results
