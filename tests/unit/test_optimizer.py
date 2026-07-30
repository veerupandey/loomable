"""Tests for the Optimizer and OptimizationRule protocol (Task 10.1).

Validates:
- Req 10.1: Optimizer is opt-in and no-op when disabled
- Req 10.2: Optimizer applies rules to produce a rewritten Flow
- Req 10.7: Only individually enabled rules are applied, in fixed order
- Req 10.8: FlowPlan captures before/after when optimization is applied
"""

from __future__ import annotations

import pytest

from loomable.flow import Edge, Flow, FlowPlan, Optimizer, OptimizationRule
from loomable.flow.optimizer.rules import OptimizationRule as RuleProtocol


# ---------------------------------------------------------------------------
# Helpers: stub rules for testing
# ---------------------------------------------------------------------------


def step_a(input):
    return f"a:{input}"


def step_b(input):
    return f"b:{input}"


def step_c(input):
    return f"c:{input}"


class NoOpRule:
    """A rule that does nothing (returns the same flow unchanged)."""

    name = "noop"

    def apply(self, flow: Flow) -> Flow:
        return flow


class RemoveLastNodeRule:
    """A rule that removes the last node from the flow (for testing).

    Simulates a dead-node-elimination style rule.
    """

    name = "remove_last_node"

    def apply(self, flow: Flow) -> Flow:
        node_ids = list(flow._nodes.keys())
        if len(node_ids) <= 1:
            return flow
        # Remove the last node and any edges referencing it
        removed_id = node_ids[-1]
        new_nodes = {nid: node.runnable for nid, node in flow._nodes.items() if nid != removed_id}
        new_edges = [e for e in flow._edges if e.source != removed_id and e.target != removed_id]
        return Flow(new_nodes, edges=new_edges, engine=flow._engine, optimizer=False)


class RenameRule:
    """A rule that creates a new flow with a marker (for testing rule ordering).

    Records its application by adding a marker to the flow metadata.
    """

    def __init__(self, rule_name: str):
        self.name = rule_name
        self.applied_count = 0

    def apply(self, flow: Flow) -> Flow:
        self.applied_count += 1
        # Return a new flow (different object) to signal the rule fired
        nodes = {nid: node.runnable for nid, node in flow._nodes.items()}
        edges = list(flow._edges)
        new_flow = Flow(nodes, edges=edges, engine=flow._engine, optimizer=False)
        return new_flow


# ---------------------------------------------------------------------------
# OptimizationRule protocol conformance
# ---------------------------------------------------------------------------


class TestOptimizationRuleProtocol:
    """OptimizationRule protocol is runtime-checkable."""

    def test_noop_rule_satisfies_protocol(self):
        """A class with name and apply satisfies the protocol."""
        rule = NoOpRule()
        assert isinstance(rule, RuleProtocol)

    def test_remove_rule_satisfies_protocol(self):
        """RemoveLastNodeRule satisfies the protocol."""
        rule = RemoveLastNodeRule()
        assert isinstance(rule, RuleProtocol)

    def test_rename_rule_satisfies_protocol(self):
        """RenameRule satisfies the protocol."""
        rule = RenameRule("test")
        assert isinstance(rule, RuleProtocol)

    def test_plain_object_does_not_satisfy_protocol(self):
        """A plain object without apply/name does not satisfy the protocol."""

        class NotARule:
            pass

        assert not isinstance(NotARule(), RuleProtocol)


# ---------------------------------------------------------------------------
# Optimizer no-op when disabled (Req 10.1)
# ---------------------------------------------------------------------------


class TestOptimizerNoOp:
    """Optimizer is a no-op when not enabled — flow runs exactly as declared."""

    def test_optimizer_disabled_returns_same_flow(self):
        """When disabled, optimize() returns the original flow unchanged."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule], enabled=False)
        flow = Flow([step_a, step_b, step_c])

        result_flow, applied = optimizer.optimize(flow)

        assert result_flow is flow
        assert applied == []

    def test_optimizer_disabled_does_not_call_rules(self):
        """When disabled, rules are never applied."""
        rule = RenameRule("should_not_fire")
        optimizer = Optimizer(rules=[rule], enabled=False)
        flow = Flow([step_a, step_b])

        optimizer.optimize(flow)

        assert rule.applied_count == 0

    def test_flow_without_optimizer_explain_mirrors_original(self):
        """A Flow with optimizer=False has explain() mirroring original."""
        flow = Flow([step_a, step_b], optimizer=False)
        plan = flow.explain()

        assert plan.optimized_nodes == plan.original_nodes
        assert plan.optimized_edges == plan.original_edges
        assert plan.applied_rules == []

    def test_optimizer_no_rules_returns_unchanged(self):
        """Optimizer with empty rules list is a no-op."""
        optimizer = Optimizer(rules=[], enabled=True)
        flow = Flow([step_a, step_b])

        result_flow, applied = optimizer.optimize(flow)

        assert result_flow is flow
        assert applied == []

    def test_optimizer_enabled_setter(self):
        """The enabled property can be toggled."""
        optimizer = Optimizer(rules=[NoOpRule()], enabled=True)
        assert optimizer.enabled is True

        optimizer.enabled = False
        assert optimizer.enabled is False


# ---------------------------------------------------------------------------
# Optimizer applies rules in fixed order (Req 10.7)
# ---------------------------------------------------------------------------


class TestOptimizerAppliesRulesInOrder:
    """Optimizer applies only enabled rules in a fixed (construction) order."""

    def test_single_rule_applied(self):
        """A single enabled rule is applied."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )

        result_flow, applied = optimizer.optimize(flow)

        assert applied == ["remove_last_node"]
        # The optimized flow should have one fewer node
        assert "c" not in result_flow._nodes
        assert len(result_flow._nodes) == 2

    def test_multiple_rules_applied_in_order(self):
        """Multiple rules are applied in construction order."""
        rule1 = RenameRule("first")
        rule2 = RenameRule("second")
        optimizer = Optimizer(rules=[rule1, rule2])
        flow = Flow([step_a, step_b])

        _, applied = optimizer.optimize(flow)

        assert applied == ["first", "second"]
        assert rule1.applied_count == 1
        assert rule2.applied_count == 1

    def test_noop_rule_not_in_applied_list(self):
        """A rule that returns the same flow is not listed in applied_rules."""
        noop = NoOpRule()
        rename = RenameRule("actual_change")
        optimizer = Optimizer(rules=[noop, rename])
        flow = Flow([step_a, step_b])

        _, applied = optimizer.optimize(flow)

        # NoOpRule returns the same instance → not in applied
        assert "noop" not in applied
        assert "actual_change" in applied

    def test_rules_property_returns_all_rules(self):
        """The rules property returns all registered rules."""
        r1 = NoOpRule()
        r2 = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[r1, r2])

        assert len(optimizer.rules) == 2


# ---------------------------------------------------------------------------
# Individual rule toggling (Req 10.7)
# ---------------------------------------------------------------------------


class TestRuleToggling:
    """Each rule is individually toggleable."""

    def test_disable_rule_by_name(self):
        """A disabled rule is skipped during optimization."""
        rule1 = RenameRule("first")
        rule2 = RenameRule("second")
        optimizer = Optimizer(rules=[rule1, rule2])

        optimizer.disable_rule("first")
        flow = Flow([step_a, step_b])
        _, applied = optimizer.optimize(flow)

        # Only 'second' should fire
        assert "first" not in applied
        assert "second" in applied
        assert rule1.applied_count == 0
        assert rule2.applied_count == 1

    def test_enable_rule_after_disable(self):
        """A re-enabled rule is applied again."""
        rule = RenameRule("toggled")
        optimizer = Optimizer(rules=[rule])

        optimizer.disable_rule("toggled")
        flow = Flow([step_a, step_b])
        _, applied1 = optimizer.optimize(flow)
        assert applied1 == []

        optimizer.enable_rule("toggled")
        _, applied2 = optimizer.optimize(flow)
        assert "toggled" in applied2

    def test_is_rule_enabled_check(self):
        """is_rule_enabled correctly reports the state."""
        rule = RenameRule("check_me")
        optimizer = Optimizer(rules=[rule])

        assert optimizer.is_rule_enabled("check_me") is True
        optimizer.disable_rule("check_me")
        assert optimizer.is_rule_enabled("check_me") is False

    def test_disable_unknown_rule_raises(self):
        """Disabling a non-existent rule raises KeyError."""
        optimizer = Optimizer(rules=[NoOpRule()])

        with pytest.raises(KeyError, match="no_such_rule"):
            optimizer.disable_rule("no_such_rule")

    def test_enable_unknown_rule_raises(self):
        """Enabling a non-existent rule raises KeyError."""
        optimizer = Optimizer(rules=[NoOpRule()])

        with pytest.raises(KeyError, match="no_such_rule"):
            optimizer.enable_rule("no_such_rule")

    def test_all_rules_disabled_is_noop(self):
        """When all rules are individually disabled, optimize is a no-op."""
        rule1 = RenameRule("a")
        rule2 = RenameRule("b")
        optimizer = Optimizer(rules=[rule1, rule2])

        optimizer.disable_rule("a")
        optimizer.disable_rule("b")
        flow = Flow([step_a, step_b])
        result_flow, applied = optimizer.optimize(flow)

        assert result_flow is flow
        assert applied == []


# ---------------------------------------------------------------------------
# FlowPlan captures before/after when optimization applied (Req 10.8)
# ---------------------------------------------------------------------------


class TestFlowPlanBeforeAfter:
    """FlowPlan presents both original and rewritten plans."""

    def test_explain_shows_optimization_result(self):
        """explain() captures the optimized topology when optimizer is active."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            optimizer=optimizer,
        )

        plan = flow.explain()

        # Original should have all three nodes
        assert "a" in plan.original_nodes
        assert "b" in plan.original_nodes
        assert "c" in plan.original_nodes
        # Optimized should be missing 'c' (removed by the rule)
        assert "a" in plan.optimized_nodes
        assert "b" in plan.optimized_nodes
        assert "c" not in plan.optimized_nodes
        # Applied rules should list the rule
        assert "remove_last_node" in plan.applied_rules

    def test_explain_original_edges_preserved(self):
        """Original edges are recorded even when optimization changes them."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            optimizer=optimizer,
        )

        plan = flow.explain()

        # Original edges preserved
        assert ("a", "b") in plan.original_edges
        assert ("b", "c") in plan.original_edges
        # Optimized edges should not have b→c (c is removed)
        assert ("b", "c") not in plan.optimized_edges
        assert ("a", "b") in plan.optimized_edges

    def test_explain_no_optimizer_mirrors(self):
        """Without optimizer, optimized mirrors original."""
        flow = Flow([step_a, step_b, step_c])
        plan = flow.explain()

        assert plan.optimized_nodes == plan.original_nodes
        assert plan.optimized_edges == plan.original_edges
        assert plan.applied_rules == []

    def test_explain_optimizer_disabled_mirrors(self):
        """With a disabled optimizer, optimized mirrors original."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule], enabled=False)
        flow = Flow([step_a, step_b, step_c], optimizer=optimizer)

        plan = flow.explain()

        assert plan.optimized_nodes == plan.original_nodes
        assert plan.optimized_edges == plan.original_edges
        assert plan.applied_rules == []

    def test_plan_str_shows_applied_rules(self):
        """FlowPlan __str__ shows applied rule names."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            optimizer=optimizer,
        )

        plan = flow.explain()
        output = str(plan)

        assert "remove_last_node" in output

    @pytest.mark.asyncio
    async def test_arun_attaches_plan_with_optimization(self):
        """Flow.arun attaches a FlowPlan with optimization results in metadata."""
        rule = RemoveLastNodeRule()
        optimizer = Optimizer(rules=[rule])
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            optimizer=optimizer,
            engine="sequential",
        )

        result = await flow.arun("hello")

        plan = result.metadata["flow_plan"]
        assert isinstance(plan, FlowPlan)
        # Original has all 3 nodes
        assert "a" in plan.original_nodes
        assert "b" in plan.original_nodes
        assert "c" in plan.original_nodes
        # Optimized should be missing 'c'
        assert "c" not in plan.optimized_nodes
        assert "remove_last_node" in plan.applied_rules
        # Engine should be SequentialEngine
        assert plan.engine == "SequentialEngine"

    @pytest.mark.asyncio
    async def test_arun_no_optimizer_plan_mirrors(self):
        """Flow.arun without optimizer has plan mirroring original."""
        flow = Flow([step_a, step_b], engine="sequential")

        result = await flow.arun("hello")

        plan = result.metadata["flow_plan"]
        assert plan.optimized_nodes == plan.original_nodes
        assert plan.optimized_edges == plan.original_edges
        assert plan.applied_rules == []


# ---------------------------------------------------------------------------
# Optimizer repr
# ---------------------------------------------------------------------------


class TestOptimizerRepr:
    """Optimizer has a useful repr."""

    def test_repr_enabled(self):
        """Enabled optimizer shows state and counts."""
        optimizer = Optimizer(rules=[NoOpRule(), RemoveLastNodeRule()])
        r = repr(optimizer)
        assert "enabled" in r
        assert "rules=2" in r

    def test_repr_disabled(self):
        """Disabled optimizer shows disabled state."""
        optimizer = Optimizer(rules=[NoOpRule()], enabled=False)
        r = repr(optimizer)
        assert "disabled" in r
