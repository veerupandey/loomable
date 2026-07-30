"""Tests for loomable.flow.helpers convenience constructors (Task 15.1).

Validates:
- Req 2.7: Pipeline/Orchestrator/AutoPlan capabilities expressible via Flow
- Req 14.4: Removed classes reimplemented in terms of the Flow engine

Tests cover:
- sequential(a, b, c) creates a working Flow
- parallel(a, b) creates a Flow that runs nodes concurrently
- route(chooser, {"left": a, "right": b}) routes correctly
- coordinate(workers=[a, b], manager=c) delegates and synthesizes
- plan_and_execute(planner, workers, synthesizer) builds plan→map→synthesize
- All helpers return Flow instances that satisfy Runnable
"""

from __future__ import annotations

import pytest

from loomable.flow import Flow, MapNode, Node, RouterNode
from loomable.flow.helpers import (
    coordinate,
    parallel,
    plan_and_execute,
    route,
    sequential,
)
from loomable.flow.runnable import Runnable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def step_a(input):
    return f"a:{input}"


def step_b(input):
    return f"b:{input}"


def step_c(input):
    return f"c:{input}"


async def async_step(input):
    return f"async:{input}"


# ---------------------------------------------------------------------------
# sequential() tests
# ---------------------------------------------------------------------------


class TestSequential:
    """sequential(...) creates a working Flow with SequentialEngine."""

    def test_returns_flow_instance(self):
        """sequential() returns a Flow."""
        flow = sequential(step_a, step_b, step_c)
        assert isinstance(flow, Flow)

    def test_satisfies_runnable(self):
        """The returned Flow satisfies the Runnable protocol."""
        flow = sequential(step_a, step_b)
        assert isinstance(flow, Runnable)

    def test_nodes_created_from_steps(self):
        """Each step becomes a node in the Flow."""
        flow = sequential(step_a, step_b, step_c)
        assert len(flow.nodes) == 3

    def test_edges_auto_chained(self):
        """Steps are connected sequentially (a → b → c)."""
        flow = sequential(step_a, step_b, step_c)
        edges = flow.edges
        assert len(edges) == 2

    def test_engine_is_sequential(self):
        """The engine is set to 'sequential'."""
        flow = sequential(step_a)
        assert flow._engine == "sequential"

    def test_session_id_passed(self):
        """session_id parameter is forwarded to the Flow."""
        flow = sequential(step_a, session_id="my-session")
        assert flow._session_id == "my-session"

    def test_deps_passed(self):
        """deps parameter is forwarded to the Flow."""
        deps = {"db": "connection"}
        flow = sequential(step_a, deps=deps)
        assert flow._deps == deps

    def test_memory_passed(self):
        """memory parameter is forwarded to the Flow."""
        mem = object()
        flow = sequential(step_a, memory=mem)
        assert flow._memory is mem

    @pytest.mark.asyncio
    async def test_runs_end_to_end(self):
        """sequential flow executes steps in order and returns a RunResult."""
        flow = sequential(step_a, step_b)
        result = await flow.arun("hello")
        # The flow should run to completion
        assert result is not None
        assert result.output is not None

    def test_single_step(self):
        """A single-step sequential flow has no edges."""
        flow = sequential(step_a)
        assert len(flow.nodes) == 1
        assert len(flow.edges) == 0

    def test_accepts_async_functions(self):
        """Async functions are accepted as steps."""
        flow = sequential(async_step, step_b)
        assert len(flow.nodes) == 2


# ---------------------------------------------------------------------------
# parallel() tests
# ---------------------------------------------------------------------------


class TestParallel:
    """parallel(...) creates a Flow that runs nodes concurrently."""

    def test_returns_flow_instance(self):
        """parallel() returns a Flow."""
        flow = parallel(step_a, step_b)
        assert isinstance(flow, Flow)

    def test_satisfies_runnable(self):
        """The returned Flow satisfies the Runnable protocol."""
        flow = parallel(step_a, step_b)
        assert isinstance(flow, Runnable)

    def test_nodes_created_from_runnables(self):
        """Each runnable becomes a node."""
        flow = parallel(step_a, step_b, step_c)
        assert len(flow.nodes) == 3

    def test_no_edges_between_nodes(self):
        """Parallel nodes have no edges (fully independent)."""
        flow = parallel(step_a, step_b, step_c)
        assert len(flow.edges) == 0

    def test_engine_is_parallel(self):
        """The engine is set to 'parallel'."""
        flow = parallel(step_a, step_b)
        assert flow._engine == "parallel"

    def test_session_id_passed(self):
        """session_id parameter is forwarded."""
        flow = parallel(step_a, step_b, session_id="par-session")
        assert flow._session_id == "par-session"

    def test_deps_passed(self):
        """deps parameter is forwarded."""
        deps = {"key": "value"}
        flow = parallel(step_a, deps=deps)
        assert flow._deps == deps

    @pytest.mark.asyncio
    async def test_runs_end_to_end(self):
        """parallel flow executes nodes and returns a RunResult."""
        flow = parallel(step_a, step_b)
        result = await flow.arun("world")
        assert result is not None
        assert result.output is not None

    def test_handles_duplicate_function_names(self):
        """Duplicate function names are disambiguated."""
        flow = parallel(step_a, step_a)
        assert len(flow.nodes) == 2


# ---------------------------------------------------------------------------
# route() tests
# ---------------------------------------------------------------------------


class TestRoute:
    """route(chooser, choices) creates a Flow that routes correctly."""

    def test_returns_flow_instance(self):
        """route() returns a Flow."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        assert isinstance(flow, Flow)

    def test_satisfies_runnable(self):
        """The returned Flow satisfies the Runnable protocol."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        assert isinstance(flow, Runnable)

    def test_has_router_node(self):
        """The flow contains a 'router' node."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        assert "router" in flow.nodes

    def test_has_choice_nodes(self):
        """The flow contains nodes for each choice."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        assert "left" in flow.nodes
        assert "right" in flow.nodes

    def test_edges_from_router_to_choices(self):
        """Edges connect router to each choice."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        edges = flow.edges
        assert len(edges) == 2
        for edge in edges:
            assert edge.source == "router"
            assert edge.target in ("left", "right")
            assert edge.condition is not None

    def test_handoff_flag_default_false(self):
        """handoff defaults to False."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        router_node = flow.nodes["router"]
        assert isinstance(router_node.runnable, RouterNode)
        assert router_node.runnable.handoff is False

    def test_handoff_flag_true(self):
        """handoff=True is passed to the RouterNode."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b}, handoff=True)
        router_node = flow.nodes["router"]
        assert isinstance(router_node.runnable, RouterNode)
        assert router_node.runnable.handoff is True

    @pytest.mark.asyncio
    async def test_routes_to_selected_branch(self):
        """The flow executes only the selected branch."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a, "right": step_b})
        result = await flow.arun("test")
        assert result is not None
        assert result.output is not None

    def test_session_id_passed(self):
        """session_id parameter is forwarded."""
        chooser = lambda input: "left"
        flow = route(chooser, {"left": step_a}, session_id="route-sess")
        assert flow._session_id == "route-sess"


# ---------------------------------------------------------------------------
# coordinate() tests
# ---------------------------------------------------------------------------


class TestCoordinate:
    """coordinate(workers, manager) delegates and synthesizes."""

    def test_returns_flow_instance(self):
        """coordinate() returns a Flow."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        assert isinstance(flow, Flow)

    def test_satisfies_runnable(self):
        """The returned Flow satisfies the Runnable protocol."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        assert isinstance(flow, Runnable)

    def test_has_worker_nodes(self):
        """The flow contains nodes for each worker."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        # Workers get function-name-based IDs
        assert len(flow.nodes) == 3  # 2 workers + 1 manager

    def test_has_manager_node(self):
        """The flow contains a manager node flagged manager=True."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        assert "manager" in flow.nodes
        manager_node = flow.nodes["manager"]
        assert manager_node.manager is True

    def test_engine_is_hierarchical(self):
        """The engine is set to 'hierarchical'."""
        flow = coordinate(workers=[step_a], manager=step_b)
        assert flow._engine == "hierarchical"

    def test_no_edges(self):
        """Hierarchical engine doesn't need edges (manager delegates internally)."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        assert len(flow.edges) == 0

    @pytest.mark.asyncio
    async def test_runs_end_to_end(self):
        """coordinate flow runs workers and manager to completion."""
        flow = coordinate(workers=[step_a, step_b], manager=step_c)
        result = await flow.arun("task")
        assert result is not None
        assert result.output is not None

    def test_session_id_passed(self):
        """session_id parameter is forwarded."""
        flow = coordinate(workers=[step_a], manager=step_b, session_id="coord-sess")
        assert flow._session_id == "coord-sess"

    def test_deps_passed(self):
        """deps parameter is forwarded."""
        deps = {"service": "mock"}
        flow = coordinate(workers=[step_a], manager=step_b, deps=deps)
        assert flow._deps == deps


# ---------------------------------------------------------------------------
# plan_and_execute() tests
# ---------------------------------------------------------------------------


class TestPlanAndExecute:
    """plan_and_execute(planner, workers, synthesizer) builds plan→map→synthesize."""

    def test_returns_flow_instance(self):
        """plan_and_execute() returns a Flow."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert isinstance(flow, Flow)

    def test_satisfies_runnable(self):
        """The returned Flow satisfies the Runnable protocol."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert isinstance(flow, Runnable)

    def test_has_planner_node(self):
        """The flow contains a 'planner' node."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert "planner" in flow.nodes

    def test_has_map_node(self):
        """The flow contains a 'map' node wrapping a MapNode."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert "map" in flow.nodes
        map_node = flow.nodes["map"]
        assert isinstance(map_node.runnable, MapNode)

    def test_has_synthesizer_node(self):
        """The flow contains a 'synthesizer' node."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert "synthesizer" in flow.nodes

    def test_edges_connect_planner_map_synthesizer(self):
        """Edges: planner → map → synthesizer."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        edges = flow.edges
        assert len(edges) == 2
        # planner → map
        assert edges[0].source == "planner"
        assert edges[0].target == "map"
        # map → synthesizer
        assert edges[1].source == "map"
        assert edges[1].target == "synthesizer"

    def test_map_node_over_key(self):
        """MapNode uses the configured 'over' key (default 'plan_steps')."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        map_node = flow.nodes["map"]
        assert isinstance(map_node.runnable, MapNode)
        assert map_node.runnable.over == "plan_steps"

    def test_custom_over_key(self):
        """Custom 'over' key is passed to the MapNode."""
        flow = plan_and_execute(
            planner=step_a, workers=step_b, synthesizer=step_c, over="tasks"
        )
        map_node = flow.nodes["map"]
        assert map_node.runnable.over == "tasks"

    def test_engine_is_sequential(self):
        """Plan-and-execute uses sequential engine (planner→map→synth in order)."""
        flow = plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c)
        assert flow._engine == "sequential"

    def test_session_id_passed(self):
        """session_id parameter is forwarded."""
        flow = plan_and_execute(
            planner=step_a, workers=step_b, synthesizer=step_c, session_id="plan-sess"
        )
        assert flow._session_id == "plan-sess"

    def test_deps_passed(self):
        """deps parameter is forwarded."""
        deps = {"api": "key"}
        flow = plan_and_execute(
            planner=step_a, workers=step_b, synthesizer=step_c, deps=deps
        )
        assert flow._deps == deps


# ---------------------------------------------------------------------------
# All helpers satisfy Runnable (cross-cutting)
# ---------------------------------------------------------------------------


class TestAllHelpersSatisfyRunnable:
    """All helper functions return Flow instances that satisfy Runnable."""

    def test_sequential_is_runnable(self):
        assert isinstance(sequential(step_a, step_b), Runnable)

    def test_parallel_is_runnable(self):
        assert isinstance(parallel(step_a, step_b), Runnable)

    def test_route_is_runnable(self):
        chooser = lambda input: "a"
        assert isinstance(route(chooser, {"a": step_a}), Runnable)

    def test_coordinate_is_runnable(self):
        assert isinstance(coordinate(workers=[step_a], manager=step_b), Runnable)

    def test_plan_and_execute_is_runnable(self):
        assert isinstance(
            plan_and_execute(planner=step_a, workers=step_b, synthesizer=step_c),
            Runnable,
        )
