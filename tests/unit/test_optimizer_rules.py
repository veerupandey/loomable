"""Tests for ParallelizeRule and DeadNodeEliminationRule (Task 10.2).

Validates:
- Req 10.3: ParallelizeRule identifies independent sequential nodes and removes
  ordering edges so the engine can run them concurrently.
- Req 10.4: DeadNodeEliminationRule removes nodes whose outputs are never
  consumed by any reachable node or the final output.
"""

from __future__ import annotations

import pytest

from loomable.flow import Edge, Flow
from loomable.flow.optimizer.rules import (
    DeadNodeEliminationRule,
    OptimizationRule,
    ParallelizeRule,
)


# ---------------------------------------------------------------------------
# Helpers: simple function nodes for testing
# ---------------------------------------------------------------------------


def step_a(input):
    return f"a:{input}"


def step_b(input):
    return f"b:{input}"


def step_c(input):
    return f"c:{input}"


def step_d(input):
    return f"d:{input}"


def step_e(input):
    return f"e:{input}"


# ---------------------------------------------------------------------------
# ParallelizeRule protocol conformance
# ---------------------------------------------------------------------------


class TestParallelizeRuleProtocol:
    """ParallelizeRule satisfies the OptimizationRule protocol."""

    def test_satisfies_protocol(self):
        rule = ParallelizeRule()
        assert isinstance(rule, OptimizationRule)

    def test_has_name(self):
        rule = ParallelizeRule()
        assert rule.name == "parallelize"


# ---------------------------------------------------------------------------
# ParallelizeRule: removes unnecessary ordering edges
# ---------------------------------------------------------------------------


class TestParallelizeRuleRemovesEdges:
    """ParallelizeRule removes edges between independent nodes."""

    def test_removes_edge_between_independent_roots(self):
        """Two nodes connected by an unconditional edge where both are roots
        (source has no predecessors) — the edge is pure ordering."""
        flow = Flow(
            {"a": step_a, "b": step_b},
            edges=[Edge(source="a", target="b")],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        # Should return a new flow with the edge removed
        assert result is not flow
        assert len(result._edges) == 0
        # Both nodes still present
        assert "a" in result._nodes
        assert "b" in result._nodes

    def test_removes_edge_when_target_has_other_predecessors(self):
        """A→C and B→C: the A→C edge is removable because C has another pred B."""
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[
                Edge(source="a", target="c"),
                Edge(source="b", target="c"),
            ],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        # Both edges A→C and B→C should be removed:
        # A→C: C has other pred B, so removable
        # B→C: C has other pred A, so removable
        assert result is not flow
        assert len(result._edges) == 0

    def test_preserves_conditional_edges(self):
        """Edges with conditions are never removed (may encode data dependency)."""
        flow = Flow(
            {"a": step_a, "b": step_b},
            edges=[Edge(source="a", target="b", condition=lambda s: True)],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        # Conditional edge preserved — flow unchanged
        assert result is flow

    def test_preserves_necessary_edge_in_chain(self):
        """In A→B→C, B→C is necessary because B is C's only predecessor and
        B is not a root (it depends on A)."""
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[
                Edge(source="a", target="b"),
                Edge(source="b", target="c"),
            ],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        # A→B: B has no other preds, but A is a root → removable
        # B→C: C has no other preds, and B is NOT a root (it has pred A) → not removable
        assert result is not flow
        # Only A→B removed; B→C preserved
        remaining_edges = [(e.source, e.target) for e in result._edges]
        assert ("a", "b") not in remaining_edges
        assert ("b", "c") in remaining_edges

    def test_no_change_returns_same_instance(self):
        """When no optimization applies, returns the same flow object."""
        # Single node, no edges
        flow = Flow({"a": step_a}, edges=[])
        rule = ParallelizeRule()
        result = rule.apply(flow)

        assert result is flow

    def test_no_change_for_necessary_chain(self):
        """A→B where B depends on A (A is not a root... wait, A IS a root).
        Actually in A→B where A is a root and B has no other preds, A→B is removable.
        Let's use a deeper chain to test preservation: X→A→B where B only has A as pred
        and A is not a root."""
        flow = Flow(
            {"x": step_a, "a": step_b, "b": step_c},
            edges=[
                Edge(source="x", target="a"),
                Edge(source="a", target="b"),
            ],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        # X→A: A has no other preds, X is a root → removable
        # A→B: B has no other preds, A is NOT a root (pred X) → NOT removable
        assert result is not flow
        remaining_edges = [(e.source, e.target) for e in result._edges]
        assert ("x", "a") not in remaining_edges
        assert ("a", "b") in remaining_edges

    def test_empty_flow_no_change(self):
        """A flow with no edges is unchanged."""
        flow = Flow({"a": step_a, "b": step_b}, edges=[])
        rule = ParallelizeRule()
        result = rule.apply(flow)

        assert result is flow

    def test_returns_new_flow_object(self):
        """When changes are made, the returned flow is a different object."""
        flow = Flow(
            {"a": step_a, "b": step_b},
            edges=[Edge(source="a", target="b")],
        )
        rule = ParallelizeRule()
        result = rule.apply(flow)

        assert result is not flow
        assert isinstance(result, Flow)


# ---------------------------------------------------------------------------
# DeadNodeEliminationRule protocol conformance
# ---------------------------------------------------------------------------


class TestDeadNodeEliminationRuleProtocol:
    """DeadNodeEliminationRule satisfies the OptimizationRule protocol."""

    def test_satisfies_protocol(self):
        rule = DeadNodeEliminationRule()
        assert isinstance(rule, OptimizationRule)

    def test_has_name(self):
        rule = DeadNodeEliminationRule()
        assert rule.name == "dead_node_elimination"


# ---------------------------------------------------------------------------
# DeadNodeEliminationRule: removes unconsumed nodes
# ---------------------------------------------------------------------------


class TestDeadNodeEliminationRuleRemovesNodes:
    """DeadNodeEliminationRule removes dead (unconsumed) nodes."""

    def test_removes_node_whose_output_goes_nowhere(self):
        """A node whose output is never consumed (no outgoing edges, not final) is dead.

        Graph: A→B with C disconnected. Topo: [a, b, c]. Final = c.
        B has no outgoing edges and is not the final node → dead.
        C is the final node → preserved.
        """
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # Topo: [a, b, c]. Final = c. 
        # B: no outgoing, not final → dead.
        # C: no outgoing, IS final → preserved.
        assert result is not flow
        assert "b" not in result._nodes
        assert "a" in result._nodes
        assert "c" in result._nodes

    def test_removes_disconnected_node(self):
        """A disconnected node (no edges at all) that isn't the topo-last is dead."""
        # A→C is the main chain. B is disconnected.
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="c")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # Topo: roots are a, b (in-degree 0). Sorted: [a, b].
        # Pop a → add c. Queue: [b, c]. Pop b. Queue: [c]. Pop c.
        # Topo: [a, b, c]. Final = c.
        # B has no outgoing edges, not final → dead.
        assert result is not flow
        assert "b" not in result._nodes
        assert "a" in result._nodes
        assert "c" in result._nodes

    def test_preserves_final_node(self):
        """The topologically-last node is never removed (it's the final output)."""
        # Simple chain A→B→C. C is last. Only C has no outgoing edges.
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # No dead nodes: A→B (B has outgoing to C), B→C (C is final).
        # All nodes have outgoing except C, and C is the final node.
        assert result is flow

    def test_removes_branch_dead_end(self):
        """A branch that goes nowhere (dead end not at the final node) is removed.

        Graph: A→B→C, A→D where D has no further edges. D is dead.
        Topo order: a first (root), then b and d (both depend on a), then c.
        Sorted roots: [a]. Pop a → add b, d. Queue: [b, d]. Pop b → add c. 
        Queue: [c, d]. Pop c. Queue: [d]. Pop d. Topo: [a, b, c, d]. Final = d.
        Hmm, d would be the last node... Let me reconsider.

        Actually let's use: A→B→D, A→C where C is the dead end.
        Topo: roots [a]. Pop a → add b, c. Queue [b, c]. Pop b → add d.
        Queue [c, d]. Pop c. Queue [d]. Pop d. Topo: [a, b, c, d]. Final = d.
        C has no outgoing edges, not final → dead.
        """
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c, "d": step_d},
            edges=[
                Edge(source="a", target="b"),
                Edge(source="b", target="d"),
                Edge(source="a", target="c"),
            ],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # C has no outgoing edges and is not the final node (d is) → dead
        assert result is not flow
        assert "c" not in result._nodes
        assert "a" in result._nodes
        assert "b" in result._nodes
        assert "d" in result._nodes
        # Edge A→C should be removed
        remaining_edges = [(e.source, e.target) for e in result._edges]
        assert ("a", "c") not in remaining_edges
        assert ("a", "b") in remaining_edges
        assert ("b", "d") in remaining_edges

    def test_no_change_returns_same_instance(self):
        """When no dead nodes exist, returns the same flow object."""
        # All nodes are consumed: A→B→C (linear chain, C is final)
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        assert result is flow

    def test_single_node_no_change(self):
        """A single-node flow is never optimized (it's trivially the final output)."""
        flow = Flow({"a": step_a}, edges=[])
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        assert result is flow

    def test_returns_new_flow_object_when_changed(self):
        """When dead nodes are removed, the result is a different Flow object."""
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="c")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # b is disconnected and not the final node → dead → new flow
        assert result is not flow
        assert isinstance(result, Flow)

    def test_removes_multiple_dead_nodes(self):
        """Multiple dead nodes are removed in one pass."""
        # A→D is the main chain. B and C are disconnected.
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c, "d": step_d},
            edges=[Edge(source="a", target="d")],
        )
        rule = DeadNodeEliminationRule()
        result = rule.apply(flow)

        # Topo: roots [a, b, c]. Pop a → add d. Queue [b, c, d].
        # Pop b. Queue [c, d]. Pop c. Queue [d]. Pop d.
        # Topo: [a, b, c, d]. Final = d.
        # B: no outgoing, not final → dead
        # C: no outgoing, not final → dead
        assert result is not flow
        assert "b" not in result._nodes
        assert "c" not in result._nodes
        assert "a" in result._nodes
        assert "d" in result._nodes


# ---------------------------------------------------------------------------
# Integration: both rules with the Optimizer
# ---------------------------------------------------------------------------


class TestRulesWithOptimizer:
    """Both rules work correctly when registered with the Optimizer."""

    def test_parallelize_rule_in_optimizer(self):
        """ParallelizeRule works when applied through the Optimizer."""
        from loomable.flow import Optimizer

        rule = ParallelizeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b},
            edges=[Edge(source="a", target="b")],
        )

        result_flow, applied = optimizer.optimize(flow)

        assert "parallelize" in applied
        assert result_flow is not flow

    def test_dce_rule_in_optimizer(self):
        """DeadNodeEliminationRule works when applied through the Optimizer."""
        from loomable.flow import Optimizer

        rule = DeadNodeEliminationRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="c")],
        )

        result_flow, applied = optimizer.optimize(flow)

        assert "dead_node_elimination" in applied
        assert "b" not in result_flow._nodes

    def test_both_rules_compose(self):
        """Both rules can be applied in sequence."""
        from loomable.flow import Optimizer

        parallelize = ParallelizeRule()
        dce = DeadNodeEliminationRule()
        optimizer = Optimizer(rules=[dce, parallelize])

        # B is dead (disconnected); A→C is an ordering edge between roots
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="c")],
        )

        result_flow, applied = optimizer.optimize(flow)

        # DCE removes B first, then parallelize may remove A→C
        assert "dead_node_elimination" in applied
