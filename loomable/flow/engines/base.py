"""ExecutionEngine protocol and shared topology utilities.

Defines the pluggable engine contract and provides toposort, cycle detection,
and level-set computation used by all concrete engines.
"""

from __future__ import annotations

__all__ = [
    "ExecutionEngine",
    "toposort",
    "detect_cycle",
    "level_sets",
]

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loomable.flow.nodes import Edge, FlowConfigError, Node

if TYPE_CHECKING:
    from loomable.agent.context import RunContext
    from loomable.agent.run import RunResult
    from loomable.flow.flow import Flow
    from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# ExecutionEngine Protocol (Req 8.1)
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionEngine(Protocol):
    """Pluggable physical-plan executor that drives a Flow to completion.

    Implementations receive the Flow (logical plan), an initial input, the
    SharedState instance, and the RunContext, and must return a RunResult.
    """

    async def run(
        self,
        flow: "Flow",
        input: Any,  # noqa: A002
        state: "SharedState",
        context: "RunContext",
    ) -> "RunResult": ...


# ---------------------------------------------------------------------------
# Shared topology utilities
# ---------------------------------------------------------------------------


def _build_adjacency(
    nodes: dict[str, Node], edges: list[Edge]
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Build adjacency list and in-degree map from nodes and edges.

    Returns
    -------
    adj : dict mapping each node_id to its list of successor node_ids.
    in_degree : dict mapping each node_id to the number of incoming edges.
    """
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}

    for edge in edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    return adj, in_degree


def toposort(nodes: dict[str, Node], edges: list[Edge]) -> list[str]:
    """Return node_ids in topological order (Kahn's algorithm).

    Parameters
    ----------
    nodes:
        Mapping of node_id to Node instances.
    edges:
        List of Edge instances defining directed dependencies.

    Returns
    -------
    A list of node_ids in a valid topological order.

    Raises
    ------
    FlowConfigError
        If the graph contains a cycle (use :func:`detect_cycle` first for
        a more descriptive error message).
    """
    adj, in_degree = _build_adjacency(nodes, edges)

    # Start with all nodes that have no incoming edges
    queue: list[str] = sorted(
        nid for nid, deg in in_degree.items() if deg == 0
    )
    result: list[str] = []

    while queue:
        # Pop the lexicographically smallest for deterministic order
        current = queue.pop(0)
        result.append(current)

        for neighbor in sorted(adj[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        # Keep queue sorted for determinism
        queue.sort()

    if len(result) != len(nodes):
        raise FlowConfigError(
            "Flow graph contains a cycle and cannot be topologically sorted."
        )

    return result


def _is_loop_node(node: Node) -> bool:
    """Check whether a node wraps a Loop (i.e. is a LoopNode).

    A LoopNode is an explicit Loop used as a node inside a Flow. Cycles
    involving such nodes are permitted (Req 8.6) because the Loop handles
    its own iteration internally.
    """
    from loomable.flow.loop import Loop

    return isinstance(node.runnable, Loop)


def detect_cycle(nodes: dict[str, Node], edges: list[Edge]) -> None:
    """Raise FlowConfigError naming the cycle unless it is an explicit LoopNode.

    Uses DFS-based cycle detection. If a cycle is found and all nodes in
    the cycle are explicit LoopNodes, the cycle is permitted. Otherwise,
    raises FlowConfigError identifying the nodes forming the cycle (Req 8.6).

    Parameters
    ----------
    nodes:
        Mapping of node_id to Node instances.
    edges:
        List of Edge instances defining directed dependencies.

    Raises
    ------
    FlowConfigError
        If a cycle is detected among non-LoopNode nodes, with an error
        message naming the nodes in the cycle.
    """
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        adj[edge.source].append(edge.target)

    # DFS states: 0 = unvisited, 1 = in current path, 2 = fully processed
    state: dict[str, int] = {nid: 0 for nid in nodes}
    # Track path for cycle reconstruction
    parent: dict[str, str | None] = {nid: None for nid in nodes}

    def _dfs(node_id: str, path: list[str]) -> list[str] | None:
        """Return the cycle (list of node_ids) if found, else None."""
        state[node_id] = 1
        path.append(node_id)

        for neighbor in sorted(adj[node_id]):
            if state[neighbor] == 1:
                # Found a back-edge: extract the cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                return cycle
            elif state[neighbor] == 0:
                result = _dfs(neighbor, path)
                if result is not None:
                    return result

        path.pop()
        state[node_id] = 2
        return None

    # Check all nodes (graph may be disconnected)
    for node_id in sorted(nodes):
        if state[node_id] == 0:
            cycle = _dfs(node_id, [])
            if cycle is not None:
                # Check if ALL nodes in the cycle are LoopNodes
                all_loop_nodes = all(
                    _is_loop_node(nodes[nid]) for nid in cycle
                )
                if not all_loop_nodes:
                    cycle_str = " -> ".join(cycle + [cycle[0]])
                    raise FlowConfigError(
                        f"Flow contains a cycle: {cycle_str}. "
                        f"Cycles are only permitted for explicit LoopNodes."
                    )


def level_sets(nodes: dict[str, Node], edges: list[Edge]) -> list[list[str]]:
    """Group nodes into dependency levels (supersteps for the Parallel engine).

    Each level contains nodes whose dependencies are all satisfied by
    previous levels. Nodes within the same level can execute concurrently.

    Parameters
    ----------
    nodes:
        Mapping of node_id to Node instances.
    edges:
        List of Edge instances defining directed dependencies.

    Returns
    -------
    A list of levels, where each level is a sorted list of node_ids that
    can execute concurrently (all their predecessors are in earlier levels).

    Raises
    ------
    FlowConfigError
        If the graph contains a cycle (all nodes cannot be assigned to levels).
    """
    adj, in_degree = _build_adjacency(nodes, edges)

    # Start with all nodes that have no incoming edges (level 0)
    current_level = sorted(nid for nid, deg in in_degree.items() if deg == 0)
    levels: list[list[str]] = []
    processed = 0

    while current_level:
        levels.append(current_level)
        processed += len(current_level)
        next_level_set: set[str] = set()

        for nid in current_level:
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_level_set.add(neighbor)

        current_level = sorted(next_level_set)

    if processed != len(nodes):
        raise FlowConfigError(
            "Flow graph contains a cycle and cannot be partitioned into levels."
        )

    return levels
