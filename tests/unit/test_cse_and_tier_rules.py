"""Tests for CommonSubexpressionRule and ModelTierRule (Task 10.3).

Validates:
- Req 10.5: CommonSubexpressionRule runs a node once and reuses its result
  when two or more nodes have identical Runnable and identical inputs.
- Req 10.6: ModelTierRule assigns lower-cost tiers to nodes flagged
  low-complexity.
"""

from __future__ import annotations

import pytest

from loomable.flow import Edge, Flow
from loomable.flow.optimizer.rules import (
    CommonSubexpressionRule,
    ModelTierRule,
    OptimizationRule,
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


# ---------------------------------------------------------------------------
# CommonSubexpressionRule protocol conformance
# ---------------------------------------------------------------------------


class TestCommonSubexpressionRuleProtocol:
    """CommonSubexpressionRule satisfies the OptimizationRule protocol."""

    def test_satisfies_protocol(self):
        rule = CommonSubexpressionRule()
        assert isinstance(rule, OptimizationRule)

    def test_has_name(self):
        rule = CommonSubexpressionRule()
        assert rule.name == "common_subexpression"


# ---------------------------------------------------------------------------
# CommonSubexpressionRule: merges identical runnables with same predecessors
# ---------------------------------------------------------------------------


class TestCommonSubexpressionRuleMerges:
    """CommonSubexpressionRule merges nodes with identical runnables and predecessors."""

    def test_merges_identical_runnables_same_predecessors(self):
        """Two nodes wrapping the same Runnable object with the same predecessors
        are merged: the second is removed and dependents redirected."""
        from loomable.flow.runnable import FunctionRunnable

        shared_runnable = FunctionRunnable(step_a)

        # Build a flow where node_1 and node_2 both use the same runnable
        # and both have the same predecessor (root, i.e., no predecessors)
        flow = Flow(
            {"node_1": shared_runnable, "node_2": shared_runnable, "sink": step_c},
            edges=[
                Edge(source="node_1", target="sink"),
                Edge(source="node_2", target="sink"),
            ],
        )

        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        # One of the duplicates should be removed
        assert result is not flow
        # Only one of node_1/node_2 should remain
        remaining = {"node_1", "node_2"} & set(result._nodes.keys())
        assert len(remaining) == 1
        # Sink should still be present
        assert "sink" in result._nodes
        # The remaining node should have an edge to sink
        edge_targets = {e.target for e in result._edges}
        assert "sink" in edge_targets

    def test_merges_identical_runnables_with_shared_predecessor(self):
        """Two nodes with the same Runnable and same predecessor are merged."""
        from loomable.flow.runnable import FunctionRunnable

        shared_runnable = FunctionRunnable(step_b)

        # root → node_1 → sink
        # root → node_2 → sink
        # node_1 and node_2 have the same runnable and same predecessor (root)
        flow = Flow(
            {"root": step_a, "node_1": shared_runnable, "node_2": shared_runnable, "sink": step_c},
            edges=[
                Edge(source="root", target="node_1"),
                Edge(source="root", target="node_2"),
                Edge(source="node_1", target="sink"),
                Edge(source="node_2", target="sink"),
            ],
        )

        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        # One of node_1/node_2 should be removed
        assert result is not flow
        remaining = {"node_1", "node_2"} & set(result._nodes.keys())
        assert len(remaining) == 1
        assert "root" in result._nodes
        assert "sink" in result._nodes

    def test_does_not_merge_different_runnables(self):
        """Nodes with different Runnable objects are not merged."""
        from loomable.flow.runnable import FunctionRunnable

        runnable_a = FunctionRunnable(step_a)
        runnable_b = FunctionRunnable(step_b)

        flow = Flow(
            {"node_1": runnable_a, "node_2": runnable_b, "sink": step_c},
            edges=[
                Edge(source="node_1", target="sink"),
                Edge(source="node_2", target="sink"),
            ],
        )

        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        # No merge: different runnables
        assert result is flow

    def test_does_not_merge_same_runnable_different_predecessors(self):
        """Same Runnable but different predecessors → no merge (different inputs)."""
        from loomable.flow.runnable import FunctionRunnable

        shared_runnable = FunctionRunnable(step_a)

        # node_1 has predecessor "root_a", node_2 has predecessor "root_b"
        flow = Flow(
            {
                "root_a": step_a,
                "root_b": step_b,
                "node_1": shared_runnable,
                "node_2": shared_runnable,
                "sink": step_c,
            },
            edges=[
                Edge(source="root_a", target="node_1"),
                Edge(source="root_b", target="node_2"),
                Edge(source="node_1", target="sink"),
                Edge(source="node_2", target="sink"),
            ],
        )

        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        # No merge: different predecessors
        assert result is flow

    def test_no_change_returns_same_instance(self):
        """When no CSE applies, returns the same flow object."""
        flow = Flow(
            {"a": step_a, "b": step_b},
            edges=[Edge(source="a", target="b")],
        )
        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        assert result is flow

    def test_single_node_no_change(self):
        """A single-node flow has nothing to merge."""
        flow = Flow({"a": step_a}, edges=[])
        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        assert result is flow

    def test_redirects_downstream_edges(self):
        """When a duplicate is removed, downstream edges are redirected to the canonical."""
        from loomable.flow.runnable import FunctionRunnable

        shared_runnable = FunctionRunnable(step_a)

        # node_1 and node_2 are CSE; downstream_1 depends on node_1, downstream_2 on node_2
        flow = Flow(
            {
                "node_1": shared_runnable,
                "node_2": shared_runnable,
                "downstream_1": step_b,
                "downstream_2": step_c,
            },
            edges=[
                Edge(source="node_1", target="downstream_1"),
                Edge(source="node_2", target="downstream_2"),
            ],
        )

        rule = CommonSubexpressionRule()
        result = rule.apply(flow)

        assert result is not flow
        # One of node_1/node_2 removed
        remaining_cse = {"node_1", "node_2"} & set(result._nodes.keys())
        assert len(remaining_cse) == 1
        canonical = remaining_cse.pop()

        # Both downstream nodes should have edges from the canonical node
        edge_sources = {e.source for e in result._edges}
        assert canonical in edge_sources
        # Both downstreams should still exist
        assert "downstream_1" in result._nodes
        assert "downstream_2" in result._nodes


# ---------------------------------------------------------------------------
# ModelTierRule protocol conformance
# ---------------------------------------------------------------------------


class TestModelTierRuleProtocol:
    """ModelTierRule satisfies the OptimizationRule protocol."""

    def test_satisfies_protocol(self):
        rule = ModelTierRule()
        assert isinstance(rule, OptimizationRule)

    def test_has_name(self):
        rule = ModelTierRule()
        assert rule.name == "model_tier"


# ---------------------------------------------------------------------------
# ModelTierRule: marks low-complexity nodes
# ---------------------------------------------------------------------------


class TestModelTierRuleMarksNodes:
    """ModelTierRule marks low-complexity nodes for cheaper model tiers."""

    def test_marks_node_with_complexity_attribute(self):
        """A runnable with complexity='low' gets marked."""
        from loomable.flow.runnable import FunctionRunnable

        runnable = FunctionRunnable(step_a)
        runnable.complexity = "low"  # type: ignore[attr-defined]

        flow = Flow({"worker": runnable, "sink": step_b}, edges=[Edge(source="worker", target="sink")])

        rule = ModelTierRule()
        result = rule.apply(flow)

        assert result is not flow
        # The node should have a model_tier attribute set to "low"
        worker_node = result._nodes["worker"]
        assert getattr(worker_node, "model_tier", None) == "low"

    def test_marks_node_with_simple_in_name(self):
        """A node whose node_id contains 'simple' gets marked."""
        flow = Flow(
            {"simple_summarizer": step_a, "complex_analyzer": step_b},
            edges=[Edge(source="simple_summarizer", target="complex_analyzer")],
        )

        rule = ModelTierRule()
        result = rule.apply(flow)

        assert result is not flow
        # simple_summarizer should be marked
        simple_node = result._nodes["simple_summarizer"]
        assert getattr(simple_node, "model_tier", None) == "low"
        # complex_analyzer should NOT be marked
        complex_node = result._nodes["complex_analyzer"]
        assert not hasattr(complex_node, "model_tier")

    def test_no_change_when_no_low_complexity(self):
        """When no nodes are low-complexity, returns the same flow."""
        flow = Flow(
            {"worker_a": step_a, "worker_b": step_b},
            edges=[Edge(source="worker_a", target="worker_b")],
        )

        rule = ModelTierRule()
        result = rule.apply(flow)

        assert result is flow

    def test_marks_multiple_low_complexity_nodes(self):
        """Multiple low-complexity nodes are all marked."""
        from loomable.flow.runnable import FunctionRunnable

        runnable_low = FunctionRunnable(step_a)
        runnable_low.complexity = "low"  # type: ignore[attr-defined]

        flow = Flow(
            {"simple_task": step_b, "low_worker": runnable_low, "heavy_worker": step_c},
            edges=[
                Edge(source="simple_task", target="heavy_worker"),
                Edge(source="low_worker", target="heavy_worker"),
            ],
        )

        rule = ModelTierRule()
        result = rule.apply(flow)

        assert result is not flow
        # Both should be marked
        assert getattr(result._nodes["simple_task"], "model_tier", None) == "low"
        assert getattr(result._nodes["low_worker"], "model_tier", None) == "low"
        # heavy_worker should not be marked
        assert not hasattr(result._nodes["heavy_worker"], "model_tier")

    def test_does_not_mark_high_complexity(self):
        """A runnable with complexity='high' is not marked."""
        from loomable.flow.runnable import FunctionRunnable

        runnable = FunctionRunnable(step_a)
        runnable.complexity = "high"  # type: ignore[attr-defined]

        flow = Flow({"worker": runnable}, edges=[])

        rule = ModelTierRule()
        result = rule.apply(flow)

        assert result is flow


# ---------------------------------------------------------------------------
# Integration: both rules with the Optimizer
# ---------------------------------------------------------------------------


class TestNewRulesWithOptimizer:
    """Both new rules work correctly when registered with the Optimizer."""

    def test_cse_rule_in_optimizer(self):
        """CommonSubexpressionRule works when applied through the Optimizer."""
        from loomable.flow.optimizer import Optimizer
        from loomable.flow.runnable import FunctionRunnable

        shared_runnable = FunctionRunnable(step_a)
        rule = CommonSubexpressionRule()
        optimizer = Optimizer(rules=[rule])

        flow = Flow(
            {"node_1": shared_runnable, "node_2": shared_runnable, "sink": step_b},
            edges=[
                Edge(source="node_1", target="sink"),
                Edge(source="node_2", target="sink"),
            ],
        )

        result_flow, applied = optimizer.optimize(flow)

        assert "common_subexpression" in applied
        assert result_flow is not flow

    def test_model_tier_rule_in_optimizer(self):
        """ModelTierRule works when applied through the Optimizer."""
        from loomable.flow.optimizer import Optimizer

        rule = ModelTierRule()
        optimizer = Optimizer(rules=[rule])

        flow = Flow(
            {"simple_task": step_a, "hard_task": step_b},
            edges=[Edge(source="simple_task", target="hard_task")],
        )

        result_flow, applied = optimizer.optimize(flow)

        assert "model_tier" in applied
        assert result_flow is not flow

    def test_all_four_rules_compose(self):
        """All four shipped rules can be applied in sequence."""
        from loomable.flow.optimizer import Optimizer
        from loomable.flow.optimizer.rules import DeadNodeEliminationRule, ParallelizeRule

        optimizer = Optimizer(rules=[
            DeadNodeEliminationRule(),
            ParallelizeRule(),
            CommonSubexpressionRule(),
            ModelTierRule(),
        ])

        flow = Flow(
            {"simple_a": step_a, "worker_b": step_b, "sink": step_c},
            edges=[
                Edge(source="simple_a", target="sink"),
                Edge(source="worker_b", target="sink"),
            ],
        )

        result_flow, applied = optimizer.optimize(flow)

        # At minimum, ModelTierRule should fire (simple_a has "simple" in name)
        assert "model_tier" in applied
