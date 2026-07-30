"""Tests for EngineSelector (automatic engine selection from topology).

Validates:
- Req 9.1: Engine_Selector analyzes topology and selects an engine.
- Req 9.2: Linear chain → SequentialEngine.
- Req 9.3: Independent branches (≥2 nodes with no path between them) → ParallelEngine.
- Req 9.4: Manager node present → HierarchicalEngine.
- Req 9.5: Explicit engine bypasses the selector.
- Req 9.6: Selected engine recorded in FlowPlan.
"""

from __future__ import annotations

import pytest

from loomable.flow.engines.hierarchical import HierarchicalEngine
from loomable.flow.engines.parallel import ParallelEngine
from loomable.flow.engines.selector import EngineSelector
from loomable.flow.engines.sequential import SequentialEngine
from loomable.flow.nodes import Edge, Node
from loomable.flow.runnable import FunctionRunnable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(input):
    return input


def _make_node(node_id: str, *, manager: bool = False) -> Node:
    """Create a Node wrapping a no-op function."""
    return Node(node_id=node_id, runnable=FunctionRunnable(_noop), manager=manager)


# ---------------------------------------------------------------------------
# Test: Linear chain selects Sequential (Req 9.2)
# ---------------------------------------------------------------------------


class TestLinearChainSelectsSequential:
    """A single linear chain topology selects SequentialEngine."""

    def test_single_node_selects_sequential(self):
        """A single-node graph is a trivial chain → Sequential."""
        nodes = {"a": _make_node("a")}
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, SequentialEngine)

    def test_two_node_chain_selects_sequential(self):
        """A → B chain selects Sequential."""
        nodes = {"a": _make_node("a"), "b": _make_node("b")}
        edges = [Edge(source="a", target="b")]
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, SequentialEngine)

    def test_three_node_chain_selects_sequential(self):
        """A → B → C linear chain selects Sequential."""
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b"),
            "c": _make_node("c"),
        }
        edges = [
            Edge(source="a", target="b"),
            Edge(source="b", target="c"),
        ]
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, SequentialEngine)

    def test_empty_graph_selects_sequential(self):
        """An empty graph (no nodes) selects Sequential as fallback."""
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, SequentialEngine)


# ---------------------------------------------------------------------------
# Test: Independent branches select Parallel (Req 9.3)
# ---------------------------------------------------------------------------


class TestIndependentBranchesSelectParallel:
    """Topologies with independent branches select ParallelEngine."""

    def test_two_independent_nodes_select_parallel(self):
        """Two nodes with no edges → they are in the same level → Parallel."""
        nodes = {"a": _make_node("a"), "b": _make_node("b")}
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, ParallelEngine)

    def test_diamond_dag_selects_parallel(self):
        """Diamond DAG: A → B, A → C, B → D, C → D has parallel levels."""
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b"),
            "c": _make_node("c"),
            "d": _make_node("d"),
        }
        edges = [
            Edge(source="a", target="b"),
            Edge(source="a", target="c"),
            Edge(source="b", target="d"),
            Edge(source="c", target="d"),
        ]
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, ParallelEngine)

    def test_fan_out_selects_parallel(self):
        """A fan-out (one source, multiple independent targets) → Parallel."""
        nodes = {
            "root": _make_node("root"),
            "leaf1": _make_node("leaf1"),
            "leaf2": _make_node("leaf2"),
            "leaf3": _make_node("leaf3"),
        }
        edges = [
            Edge(source="root", target="leaf1"),
            Edge(source="root", target="leaf2"),
            Edge(source="root", target="leaf3"),
        ]
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, ParallelEngine)

    def test_three_independent_nodes_select_parallel(self):
        """Three disconnected nodes → all in level 0 → Parallel."""
        nodes = {
            "x": _make_node("x"),
            "y": _make_node("y"),
            "z": _make_node("z"),
        }
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, ParallelEngine)


# ---------------------------------------------------------------------------
# Test: Manager node present selects Hierarchical (Req 9.4)
# ---------------------------------------------------------------------------


class TestManagerSelectsHierarchical:
    """A manager node triggers HierarchicalEngine selection."""

    def test_single_manager_selects_hierarchical(self):
        """One node with manager=True → Hierarchical, regardless of topology."""
        nodes = {
            "worker1": _make_node("worker1"),
            "worker2": _make_node("worker2"),
            "mgr": _make_node("mgr", manager=True),
        }
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, HierarchicalEngine)

    def test_manager_in_chain_selects_hierarchical(self):
        """Even in a chain, a manager node triggers Hierarchical."""
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b", manager=True),
        }
        edges = [Edge(source="a", target="b")]
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, HierarchicalEngine)

    def test_manager_priority_over_parallel(self):
        """Manager takes priority: even with independent branches, Hierarchical wins."""
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b"),
            "mgr": _make_node("mgr", manager=True),
        }
        # a and b are independent (would be Parallel without manager)
        edges: list[Edge] = []
        engine = EngineSelector.select(nodes, edges)
        assert isinstance(engine, HierarchicalEngine)


# ---------------------------------------------------------------------------
# Test: Explicit engine bypasses selector (Req 9.5)
# ---------------------------------------------------------------------------


class TestExplicitEngineBypassesSelector:
    """Explicit engine string bypasses the EngineSelector."""

    def test_explicit_sequential_on_parallel_topology(self):
        """engine='sequential' forces Sequential even on a parallel topology."""
        from loomable.flow.flow import Flow

        # Two independent nodes → would be Parallel with auto
        flow = Flow({"a": _noop, "b": _noop}, engine="sequential")
        engine = flow._resolve_engine()
        assert isinstance(engine, SequentialEngine)

    def test_explicit_parallel_on_linear_topology(self):
        """engine='parallel' forces Parallel even on a linear chain."""
        from loomable.flow.flow import Flow

        flow = Flow([_noop, _noop], engine="parallel")
        engine = flow._resolve_engine()
        assert isinstance(engine, ParallelEngine)

    def test_explicit_hierarchical_bypasses_auto(self):
        """engine='hierarchical' forces Hierarchical regardless of topology."""
        from loomable.flow.flow import Flow

        flow = Flow([_noop], engine="hierarchical")
        engine = flow._resolve_engine()
        assert isinstance(engine, HierarchicalEngine)

    def test_custom_engine_object_bypasses_selector(self):
        """A custom engine object is used directly without selector."""
        from loomable.flow.flow import Flow

        class MyEngine:
            async def run(self, flow, input, state, context):
                pass

        custom = MyEngine()
        flow = Flow([_noop], engine=custom)
        engine = flow._resolve_engine()
        assert engine is custom


# ---------------------------------------------------------------------------
# Test: Selection recorded in FlowPlan (Req 9.6)
# ---------------------------------------------------------------------------


class TestSelectionRecordedInFlowPlan:
    """The selected engine is recorded in the FlowPlan after a run."""

    async def test_auto_linear_records_sequential_in_plan(self):
        """engine='auto' on a linear chain records SequentialEngine in FlowPlan."""
        from loomable.flow.flow import Flow

        async def step(input):
            return f"done:{input}"

        flow = Flow([step])
        result = await flow.arun("test")

        plan = result.metadata["flow_plan"]
        assert plan.engine == "SequentialEngine"

    async def test_auto_parallel_records_parallel_in_plan(self):
        """engine='auto' on independent nodes records ParallelEngine in FlowPlan."""
        from loomable.flow.flow import Flow

        async def task_a(input):
            return f"a:{input}"

        async def task_b(input):
            return f"b:{input}"

        flow = Flow({"a": task_a, "b": task_b})
        result = await flow.arun("test")

        plan = result.metadata["flow_plan"]
        assert plan.engine == "ParallelEngine"

    async def test_auto_hierarchical_records_hierarchical_in_plan(self):
        """engine='auto' with manager records HierarchicalEngine in FlowPlan."""
        from loomable.flow.flow import Flow
        from loomable.flow.nodes import Node as FlowNode

        async def worker(input):
            return f"w:{input}"

        async def mgr(input):
            return f"mgr:{input}"

        # Build flow with a manager node
        flow = Flow.__new__(Flow)
        flow._engine = "auto"
        flow._optimizer = False
        flow._memory = None
        flow._checkpointer = None
        flow._events = None
        flow._session_id = None
        flow._deps = None
        flow._reducers = None
        flow._nodes = {
            "worker": FlowNode(node_id="worker", runnable=FunctionRunnable(worker)),
            "mgr": FlowNode(node_id="mgr", runnable=FunctionRunnable(mgr), manager=True),
        }
        flow._edges = []

        result = await flow.arun("test")

        plan = result.metadata["flow_plan"]
        assert plan.engine == "HierarchicalEngine"

    async def test_explicit_engine_records_class_name(self):
        """Explicit engine string records the resolved engine class name."""
        from loomable.flow.flow import Flow

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine="sequential")
        result = await flow.arun("test")

        plan = result.metadata["flow_plan"]
        assert plan.engine == "SequentialEngine"
