"""Advanced Flow pattern helpers (prefer ``Workflow`` / ``Team``).

Thin wrappers that build a ``Flow`` for sequential / parallel / route /
coordinate / plan-and-execute graphs. ``plan_and_execute`` is also used by
``Workflow.map``.
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
    checkpointer: Any = None,
    events: Any = None,
) -> Flow:
    """Create a Flow that runs steps sequentially.

    Prefer :class:`~loomable.flow.workflow.Workflow` for new code::

        Workflow("pipe").step("a", a).step("b", b)

    Each step's output becomes the next step's input through SharedState.
    Nodes are auto-chained in order using the SequentialEngine.
    """
    return Flow(
        list(steps),
        engine="sequential",
        session_id=session_id,
        deps=deps,
        memory=memory,
        checkpointer=checkpointer,
        events=events,
    )


def parallel(
    *runnables: Any,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
    checkpointer: Any = None,
    events: Any = None,
) -> Flow:
    """Create a Flow that runs runnables concurrently.

    Prefer ``Workflow(...).parallel(...)`` for new code.
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
        checkpointer=checkpointer,
        events=events,
    )


def route(
    chooser: Any,
    choices: dict[str, Any],
    *,
    handoff: bool = False,
    session_id: str | None = None,
    deps: Any = None,
    memory: Any = None,
    checkpointer: Any = None,
    events: Any = None,
) -> Flow:
    """Create a Flow that routes to one branch.

    Prefer ``Workflow(...).branch(when=..., then=..., else_=...)`` for new code.
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
        checkpointer=checkpointer,
        events=events,
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
    checkpointer: Any = None,
    events: Any = None,
) -> Flow:
    """Create a Flow that delegates to workers then synthesizes.

    Prefer :class:`~loomable.agent.team.Team` for LLM-driven coordination,
    or ``Workflow(...).parallel(...).step("manager", manager)`` for fixed topology.
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
        checkpointer=checkpointer,
        events=events,
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
    checkpointer: Any = None,
    events: Any = None,
) -> Flow:
    """Create a plan→map→synthesize Flow. Prefer ``Workflow(...).map(worker)``."""
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
        checkpointer=checkpointer,
        events=events,
    )
