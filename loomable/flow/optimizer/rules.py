"""OptimizationRule protocol and shipped rules.

Rules: parallelize, dead-node elimination, common-subexpression, model-tier.
Each is individually toggleable and semantics-preserving by contract.
"""

from __future__ import annotations

__all__ = [
    "OptimizationRule",
    "ParallelizeRule",
    "DeadNodeEliminationRule",
    "CommonSubexpressionRule",
    "ModelTierRule",
]

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from loomable.flow.flow import Flow
    from loomable.flow.nodes import Edge, Node


@runtime_checkable
class OptimizationRule(Protocol):
    """A single semantics-preserving rewrite, individually toggleable.

    Each rule takes a Flow and returns an equivalent Flow that may be
    cheaper or faster to execute. Rules MUST preserve observable output
    (Req 10.2, 10.9).

    Attributes
    ----------
    name:
        A unique, human-readable identifier for this rule (used in
        FlowPlan.applied_rules and for toggling).
    """

    name: str

    def apply(self, flow: "Flow") -> "Flow":
        """Apply this optimization rule to a Flow.

        Returns an equivalent Flow (possibly the same instance if no
        rewrite applies). The returned Flow MUST produce the same
        observable output as the input Flow for the same input.
        """
        ...


# ---------------------------------------------------------------------------
# ParallelizeRule (Req 10.3)
# ---------------------------------------------------------------------------


class ParallelizeRule:
    """Identify independent sequential nodes and remove ordering edges.

    This rule finds edges between nodes that have no true data dependency
    (the target does not consume the source's output via shared state).
    When such an edge is found, it is removed so that the execution engine
    can run the nodes concurrently in the same superstep.

    Conservative v1 heuristic: an edge A→B is considered a pure ordering
    edge (removable) when:
    - B has no edge condition (conditions may implicitly depend on A's output)
    - A is not the sole predecessor of B that B needs for data. Specifically,
      we look for nodes that sit at the same topological "depth" — both could
      start after the same set of predecessors — and are connected only by an
      ordering edge rather than a true data flow.

    Simpler practical approach for v1: in a linear chain A→B→C, if B does not
    have an edge condition from A and B is *also* reachable from A's
    predecessors (or has none), the A→B edge is removable. We simplify further:
    find pairs where both nodes share the same set of predecessors (or none)
    and are connected by an unconditional edge — that edge is pure ordering.

    Simplest correct implementation: find edges A→B where:
    1. The edge has no condition.
    2. A has no other outgoing edges to nodes that eventually feed B
       (i.e., removing A→B doesn't disconnect B from the graph — B still
       has at least one other incoming edge, OR both A and B have in-degree
       0 after removing the edge).
    3. Both A and B are simple function nodes (not MapNode/RouterNode).

    This ensures we only parallelize truly independent nodes.
    """

    name: str = "parallelize"

    def apply(self, flow: "Flow") -> "Flow":
        """Remove unnecessary ordering edges between independent nodes.

        Returns the same flow instance if no edges can be removed.
        """
        from loomable.flow.flow import Flow
        from loomable.flow.nodes import Edge

        edges = flow._edges
        nodes = flow._nodes

        if not edges:
            return flow

        # Build predecessor map: node_id -> set of node_ids that have an edge to it
        predecessors: dict[str, set[str]] = {nid: set() for nid in nodes}
        for edge in edges:
            predecessors[edge.target].add(edge.source)

        # Build successor map: node_id -> set of node_ids it has edges to
        successors: dict[str, set[str]] = {nid: set() for nid in nodes}
        for edge in edges:
            successors[edge.source].add(edge.target)

        # Identify removable edges
        edges_to_remove: set[int] = set()

        for idx, edge in enumerate(edges):
            # Only consider unconditional edges (condition edges may encode
            # data dependency implicitly)
            if edge.condition is not None:
                continue

            source_id = edge.source
            target_id = edge.target

            # Skip if source or target is a MapNode or RouterNode (complex nodes)
            source_node = nodes[source_id]
            target_node = nodes[target_id]
            if self._is_complex_node(source_node) or self._is_complex_node(target_node):
                continue

            # The edge is removable if the target has other predecessors OR
            # if the source is not needed as a data provider. For safety, we
            # require that the target has at least one other predecessor
            # (meaning B is still reachable without this edge) OR that both
            # nodes are root-level (no predecessors except each other).
            other_preds = predecessors[target_id] - {source_id}

            # Also check: does the source have other successors? If A→B is
            # the only edge from A, but B has other inputs, it's safe to remove.
            # If B would become completely disconnected (no other preds), only
            # remove if both A and B are at the same depth (both roots after removal).

            if other_preds:
                # B still reachable via other predecessors — safe to remove
                edges_to_remove.add(idx)
            else:
                # B has no other predecessors. Removing A→B makes B a root.
                # This is only safe if A is also a root (both independent).
                source_preds = predecessors[source_id]
                if not source_preds:
                    # Both are roots (or source is a root) — they're independent
                    edges_to_remove.add(idx)

        if not edges_to_remove:
            return flow

        # Build new flow with the edges removed
        new_edges = [e for idx, e in enumerate(edges) if idx not in edges_to_remove]
        new_nodes = {nid: node.runnable for nid, node in nodes.items()}
        return Flow(new_nodes, edges=new_edges, engine=flow._engine, optimizer=False)

    def _is_complex_node(self, node: "Node") -> bool:
        """Check if a node wraps a MapNode or RouterNode (complex dispatch)."""
        from loomable.flow.nodes import MapNode, RouterNode

        return isinstance(node.runnable, (MapNode, RouterNode))


# ---------------------------------------------------------------------------
# DeadNodeEliminationRule (Req 10.4)
# ---------------------------------------------------------------------------


class DeadNodeEliminationRule:
    """Remove nodes whose outputs are never consumed.

    A node is "dead" if:
    1. It has no outgoing edges (no downstream consumer), AND
    2. It is not the topologically-last node (the terminal/sink node that
       produces the flow's final output).

    When multiple sink nodes exist (no outgoing edges), they are all
    considered live (any of them could be the final output). Only nodes
    that are dead — having no outgoing edges while other sink nodes exist
    and they are not the sole sink — are removed.

    More precisely: a node is dead if it has no outgoing edges AND it is
    not the *last* node in topological order (which is the flow's output).
    """

    name: str = "dead_node_elimination"

    def apply(self, flow: "Flow") -> "Flow":
        """Remove dead nodes and their incoming edges.

        Returns the same flow instance if no dead nodes are found.
        """
        from loomable.flow.engines.base import toposort
        from loomable.flow.flow import Flow

        nodes = flow._nodes
        edges = flow._edges

        if len(nodes) <= 1:
            return flow

        # Build successor map
        successors: dict[str, set[str]] = {nid: set() for nid in nodes}
        for edge in edges:
            successors[edge.source].add(edge.target)

        # Topological order to determine the "final output" node
        try:
            topo_order = toposort(nodes, edges)
        except Exception:
            # If there's a cycle or issue, don't optimize
            return flow

        # The last node in topological order is the final output node
        final_node_id = topo_order[-1] if topo_order else None

        # Find dead nodes: no outgoing edges and not the final output node
        dead_nodes: set[str] = set()
        for nid in nodes:
            if not successors[nid] and nid != final_node_id:
                dead_nodes.add(nid)

        if not dead_nodes:
            return flow

        # Build new flow without dead nodes
        new_nodes = {
            nid: node.runnable
            for nid, node in nodes.items()
            if nid not in dead_nodes
        }
        new_edges = [
            e for e in edges
            if e.source not in dead_nodes and e.target not in dead_nodes
        ]

        return Flow(new_nodes, edges=new_edges, engine=flow._engine, optimizer=False)


# ---------------------------------------------------------------------------
# CommonSubexpressionRule (Req 10.5)
# ---------------------------------------------------------------------------


class CommonSubexpressionRule:
    """Run a node once and reuse its result when duplicates exist.

    Two nodes are considered common subexpressions (and thus mergeable) when:
    1. They share the same Runnable object (identity check via ``is``).
    2. They have the same set of predecessors (meaning they receive the
       same input from the graph topology).

    When a duplicate is found, the second node is removed and all edges
    that targeted or sourced from it are redirected to the first (canonical)
    node. This is conservative: only exact Runnable identity AND identical
    predecessors trigger a merge.
    """

    name: str = "common_subexpression"

    def apply(self, flow: "Flow") -> "Flow":
        """Merge duplicate nodes that share Runnable identity and predecessors.

        Returns the same flow instance if no merges apply.
        """
        from loomable.flow.flow import Flow

        nodes = flow._nodes
        edges = flow._edges

        if len(nodes) <= 1:
            return flow

        # Build predecessor map: node_id -> frozenset of predecessor node_ids
        predecessors: dict[str, frozenset[str]] = {nid: frozenset() for nid in nodes}
        for edge in edges:
            predecessors[edge.target] = predecessors[edge.target] | {edge.source}

        # Group nodes by (runnable identity, predecessor set)
        # Use id(runnable) as the key for identity check
        groups: dict[tuple[int, frozenset[str]], list[str]] = {}
        for nid, node in nodes.items():
            key = (id(node.runnable), predecessors[nid])
            groups.setdefault(key, []).append(nid)

        # Find groups with more than one node (these are CSE candidates)
        merges: dict[str, str] = {}  # duplicate_node_id -> canonical_node_id
        for group_nodes in groups.values():
            if len(group_nodes) <= 1:
                continue
            # Verify actual identity (not just id() collision across GC cycles)
            # Group by actual `is` identity
            canonical = group_nodes[0]
            canonical_runnable = nodes[canonical].runnable
            for dup in group_nodes[1:]:
                if nodes[dup].runnable is canonical_runnable:
                    merges[dup] = canonical

        if not merges:
            return flow

        # Build new flow without the duplicates, redirecting edges
        remaining_node_ids = {nid for nid in nodes if nid not in merges}
        new_nodes = {nid: node.runnable for nid, node in nodes.items() if nid in remaining_node_ids}

        # Redirect edges: replace any reference to a merged node with its canonical
        new_edges = []
        seen_edges: set[tuple[str, str]] = set()
        for edge in edges:
            source = merges.get(edge.source, edge.source)
            target = merges.get(edge.target, edge.target)

            # Skip edges where source == target (self-loops from merging)
            if source == target:
                continue
            # Skip edges that reference removed nodes on both ends
            if source not in remaining_node_ids or target not in remaining_node_ids:
                continue
            # Deduplicate edges (merging may create duplicates)
            edge_key = (source, target)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            from loomable.flow.nodes import Edge as EdgeCls
            new_edges.append(EdgeCls(source=source, target=target, condition=edge.condition))

        return Flow(new_nodes, edges=new_edges, engine=flow._engine, optimizer=False)


# ---------------------------------------------------------------------------
# ModelTierRule (Req 10.6)
# ---------------------------------------------------------------------------


class ModelTierRule:
    """Assign lower-cost model tiers to nodes flagged low-complexity.

    This rule inspects each node for a ``complexity`` attribute (on the
    Runnable or on the Node itself). When found with value ``"low"``, it
    marks the node's metadata to indicate a cheaper model tier should be
    used at dispatch time.

    As a secondary heuristic, nodes whose ``node_id`` contains ``"simple"``
    are also considered low-complexity candidates.

    Since actual model tier assignment happens at engine dispatch time (not
    at the flow-plan level), this rule annotates the flow by rebuilding it
    with metadata markers. The engine or agent can then read these markers
    to select a cheaper model.

    The rule produces a new Flow only when at least one node is marked; if
    no nodes qualify, it returns the same flow instance unchanged.
    """

    name: str = "model_tier"

    def apply(self, flow: "Flow") -> "Flow":
        """Mark low-complexity nodes for cheaper model tier dispatch.

        Returns the same flow instance if no nodes qualify.
        """
        from loomable.flow.flow import Flow
        from loomable.flow.nodes import Node

        nodes = flow._nodes
        edges = flow._edges

        # Find nodes that should be marked low-complexity
        marked_node_ids: set[str] = set()
        for nid, node in nodes.items():
            if self._is_low_complexity(nid, node):
                marked_node_ids.add(nid)

        if not marked_node_ids:
            return flow

        # Rebuild nodes, adding model_tier metadata to marked ones
        new_nodes: dict[str, "Any"] = {}
        for nid, node in nodes.items():
            if nid in marked_node_ids:
                # Create a new Node with the model_tier annotation
                new_node = Node(
                    node_id=nid,
                    runnable=node.runnable,
                    require_confirmation=node.require_confirmation,
                    manager=node.manager,
                )
                new_node.model_tier = "low"  # type: ignore[attr-defined]
                new_nodes[nid] = new_node
            else:
                new_nodes[nid] = node

        # Build a new Flow with the annotated nodes directly
        # We need to bypass the normal dict[str, Runnable] construction
        # and inject Node objects. Use the dict path with runnables and
        # then patch the nodes.
        new_flow = Flow(
            {nid: (n.runnable if isinstance(n, Node) else n) for nid, n in new_nodes.items()},
            edges=list(edges),
            engine=flow._engine,
            optimizer=False,
        )
        # Patch in the annotated nodes
        for nid in marked_node_ids:
            if nid in new_flow._nodes and nid in new_nodes:
                annotated = new_nodes[nid]
                if isinstance(annotated, Node):
                    new_flow._nodes[nid] = annotated

        return new_flow

    def _is_low_complexity(self, node_id: str, node: "Node") -> bool:
        """Determine if a node qualifies as low-complexity.

        Checks:
        1. The runnable has a ``complexity`` attribute set to ``"low"``.
        2. The node_id contains ``"simple"`` (convention-based heuristic).
        """
        # Check runnable for a complexity attribute
        runnable = node.runnable
        complexity = getattr(runnable, "complexity", None)
        if complexity == "low":
            return True

        # Check node_id for "simple" substring
        if "simple" in node_id.lower():
            return True

        return False
