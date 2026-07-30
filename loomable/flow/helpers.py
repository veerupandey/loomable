"""Convenience constructors for common Flow patterns.

These helpers replace the removed `Pipeline`, `Orchestrator`, and `AutoPlan`
classes with thin wrappers that build the equivalent `Flow` using the unified
engine model (Req 2.7, 14.4).

- ``sequential(...)`` — replaces ``Pipeline``: sequential chain via SequentialEngine.
- ``parallel(...)`` — replaces ``Orchestrator(mode=PARALLEL)``: concurrent execution.
- ``route(...)`` — replaces ``Orchestrator(mode=ROUTE)``: predicate/model routing.
- ``coordinate(...)`` — replaces ``Orchestrator(mode=COORDINATE)``: hierarchical delegation.
- ``plan_and_execute(...)`` — replaces ``AutoPlan``: plan → map → synthesize.
"""

from __future__ import annotations

__all__ = [
    "coordinate",
    "parallel",
    "plan_and_execute",
    "route",
    "sequential",
]

from typing import Any

from loomable.flow.flow import Flow
from loomable.flow.nodes import Edge, MapNode, Node, RouterNode
from loomable.flow.runnable import FunctionRunnable, Runnable


def _ensure_runnable(obj: Any) -> Runnable:
    """Wrap a plain callable into a FunctionRunnable if needed."""
    if isinstance(obj, Runnable):
        return obj
    if callable(obj):
        return FunctionRunnable(obj)
    raise TypeError(
        f"Cannot use {type(obj).__name__!r} as a runnable; "
        "it must be a Runnable or a callable."
    )


def sequential(
    *steps: Any,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
) -> Flow:
    """Create a Flow that runs steps sequentially (replaces Pipeline).

    Each step's output becomes the next step's input through SharedState.
    Nodes are auto-chained in order using the SequentialEngine.

    Parameters
    ----------
    *steps:
        Runnables (or plain callables) to execute in order.
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all steps.
    memory:
        A shared MemoryStore instance available to all steps.

    Returns
    -------
    Flow
        A Flow configured with engine="sequential".
    """
    return Flow(
        list(steps),
        engine="sequential",
        session_id=session_id,
        deps=deps,
        memory=memory,
    )


def parallel(
    *runnables: Any,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
) -> Flow:
    """Create a Flow that runs runnables concurrently (replaces Orchestrator PARALLEL).

    All runnables execute in the same superstep via the ParallelEngine.
    Since they have no edges between them, they form independent branches.

    Parameters
    ----------
    *runnables:
        Runnables (or plain callables) to execute concurrently.
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all runnables.
    memory:
        A shared MemoryStore instance available to all runnables.

    Returns
    -------
    Flow
        A Flow configured with engine="parallel" and no inter-node edges.
    """
    # Build a dict of node_id → runnable with no edges (fully independent)
    nodes: dict[str, Any] = {}
    for i, r in enumerate(runnables):
        runnable = _ensure_runnable(r)
        # Derive a node_id from the runnable
        node_id = Flow._derive_node_id(r, i)
        # Ensure uniqueness
        if node_id in nodes:
            node_id = f"{node_id}_{i}"
        nodes[node_id] = runnable

    return Flow(
        nodes,
        edges=[],
        engine="parallel",
        session_id=session_id,
        deps=deps,
        memory=memory,
    )


def route(
    chooser: Any,
    choices: dict[str, Any],
    *,
    handoff: bool = False,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
) -> Flow:
    """Create a Flow that routes to one branch (replaces Orchestrator ROUTE).

    A RouterNode evaluates the chooser to select which downstream node to
    run. Only the selected branch executes; others are skipped.

    Parameters
    ----------
    chooser:
        A Runnable or Callable that returns the selected node_id (a key
        from the choices dict).
    choices:
        A dict mapping node_id → Runnable (or callable) for each possible
        route target.
    handoff:
        When True, the selected node owns the final output.
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all nodes.
    memory:
        A shared MemoryStore instance available to all nodes.

    Returns
    -------
    Flow
        A Flow with a RouterNode connected to the choice branches.
    """
    choice_ids = list(choices.keys())

    # Build the RouterNode
    router_node = RouterNode(chooser, choices=choice_ids, handoff=handoff)

    # Build the graph: router → each choice (gated by edge condition)
    graph_nodes: dict[str, Any] = {"router": Node(node_id="router", runnable=router_node)}
    edges: list[Edge] = []

    for choice_id, runnable_or_callable in choices.items():
        runnable = _ensure_runnable(runnable_or_callable)
        graph_nodes[choice_id] = Node(node_id=choice_id, runnable=runnable)
        # Edge from router to each choice, gated by whether router selected it
        edges.append(
            Edge(
                source="router",
                target=choice_id,
                condition=_make_route_condition(choice_id),
            )
        )

    return Flow(
        graph_nodes,
        edges=edges,
        engine="sequential",
        session_id=session_id,
        deps=deps,
        memory=memory,
    )


def _make_route_condition(choice_id: str):
    """Create an edge condition that checks if the router selected this choice."""

    def _condition(state) -> bool:
        selection = state.get("_router_selection")
        if isinstance(selection, list):
            return choice_id in selection
        return selection == choice_id

    return _condition


def coordinate(
    workers: list[Any],
    manager: Any,
    *,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
) -> Flow:
    """Create a Flow that delegates to workers then synthesizes (replaces Orchestrator COORDINATE).

    Workers run concurrently, and the manager node synthesizes their results.
    Uses the HierarchicalEngine with the manager flagged as ``manager=True``.

    Parameters
    ----------
    workers:
        List of Runnables (or callables) to run as workers.
    manager:
        The Runnable (or callable) that synthesizes worker results.
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all nodes.
    memory:
        A shared MemoryStore instance available to all nodes.

    Returns
    -------
    Flow
        A Flow configured with engine="hierarchical" and a manager node.
    """
    graph_nodes: dict[str, Any] = {}

    # Add worker nodes
    for i, w in enumerate(workers):
        runnable = _ensure_runnable(w)
        node_id = Flow._derive_node_id(w, i)
        if node_id in graph_nodes:
            node_id = f"{node_id}_{i}"
        graph_nodes[node_id] = Node(node_id=node_id, runnable=runnable)

    # Add manager node (flagged manager=True)
    manager_runnable = _ensure_runnable(manager)
    graph_nodes["manager"] = Node(
        node_id="manager", runnable=manager_runnable, manager=True
    )

    return Flow(
        graph_nodes,
        edges=[],
        engine="hierarchical",
        session_id=session_id,
        deps=deps,
        memory=memory,
    )


def plan_and_execute(
    planner: Any,
    workers: Any,
    synthesizer: Any,
    *,
    over: str = "plan_steps",
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
) -> Flow:
    """Create a plan→map→synthesize Flow (replaces AutoPlan).

    The planner node produces a list of steps (written to SharedState under
    the ``over`` key). The MapNode fans out the workers runnable over those
    steps concurrently. The synthesizer node combines the map results into
    a final answer.

    Parameters
    ----------
    planner:
        A Runnable (or callable) that produces a plan. Its output is expected
        to be stored in SharedState under the ``over`` key as a list.
    workers:
        A Runnable (or callable) that processes each planned step.
    synthesizer:
        A Runnable (or callable) that combines the map results.
    over:
        The SharedState key where the planner writes the list of steps.
        Defaults to "plan_steps".
    session_id:
        Optional session identifier for memory/checkpoint scoping.
    deps:
        Typed dependency injection object shared across all nodes.
    memory:
        A shared MemoryStore instance available to all nodes.

    Returns
    -------
    Flow
        A Flow that executes plan → map → synthesize.
    """
    workers_runnable = _ensure_runnable(workers)
    map_node = MapNode(body=workers_runnable, over=over)

    graph_nodes: dict[str, Any] = {
        "planner": Node(node_id="planner", runnable=_ensure_runnable(planner)),
        "map": Node(node_id="map", runnable=map_node),
        "synthesizer": Node(node_id="synthesizer", runnable=_ensure_runnable(synthesizer)),
    }

    edges = [
        Edge(source="planner", target="map"),
        Edge(source="map", target="synthesizer"),
    ]

    return Flow(
        graph_nodes,
        edges=edges,
        engine="sequential",
        session_id=session_id,
        deps=deps,
        memory=memory,
    )
