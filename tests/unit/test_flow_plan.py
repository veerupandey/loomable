"""Tests for FlowPlan and Flow.explain() (Task 5.3).

Validates:
- Req 6.7: Flow is a Runnable (composes, nests, serves identically to an agent)
- Req 9.6: Selected engine is recorded in the FlowPlan
- Req 13.4: RunResult contains the executed FlowPlan
"""

from __future__ import annotations

from loomable.flow import Edge, Flow, FlowPlan
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


# ---------------------------------------------------------------------------
# Flow.explain() returns a FlowPlan with correct nodes and edges
# ---------------------------------------------------------------------------


class TestFlowExplain:
    """Flow.explain() returns a FlowPlan with correct topology and metadata."""

    def test_explain_returns_flow_plan_instance(self):
        """explain() returns a FlowPlan dataclass."""
        flow = Flow([step_a, step_b])
        plan = flow.explain()
        assert isinstance(plan, FlowPlan)

    def test_explain_captures_original_nodes(self):
        """FlowPlan.original_nodes lists all node_ids."""
        flow = Flow({"x": step_a, "y": step_b}, edges=[Edge(source="x", target="y")])
        plan = flow.explain()
        assert set(plan.original_nodes) == {"x", "y"}

    def test_explain_captures_original_edges(self):
        """FlowPlan.original_edges lists all (source, target) tuples."""
        flow = Flow(
            {"a": step_a, "b": step_b, "c": step_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )
        plan = flow.explain()
        assert ("a", "b") in plan.original_edges
        assert ("b", "c") in plan.original_edges
        assert len(plan.original_edges) == 2

    def test_explain_list_shorthand_nodes_and_edges(self):
        """List shorthand correctly populates the plan."""
        flow = Flow([step_a, step_b, step_c])
        plan = flow.explain()

        assert "step_a" in plan.original_nodes
        assert "step_b" in plan.original_nodes
        assert "step_c" in plan.original_nodes
        assert ("step_a", "step_b") in plan.original_edges
        assert ("step_b", "step_c") in plan.original_edges

    def test_explain_optimized_mirrors_original_pre_optimizer(self):
        """Before optimization, optimized topology mirrors original."""
        flow = Flow([step_a, step_b])
        plan = flow.explain()

        assert plan.optimized_nodes == plan.original_nodes
        assert plan.optimized_edges == plan.original_edges

    def test_explain_applied_rules_empty_pre_optimizer(self):
        """Before optimization, applied_rules is empty."""
        flow = Flow([step_a, step_b])
        plan = flow.explain()

        assert plan.applied_rules == []

    def test_explain_engine_string_auto(self):
        """Default engine='auto' is recorded."""
        flow = Flow([step_a])
        plan = flow.explain()
        assert plan.engine == "auto"

    def test_explain_engine_string_explicit(self):
        """Explicit engine string is recorded."""
        flow = Flow([step_a], engine="sequential")
        plan = flow.explain()
        assert plan.engine == "sequential"

    def test_explain_engine_custom_object(self):
        """Custom engine objects record their class name."""

        class MyCustomEngine:
            async def run(self, flow, input, state, context):
                pass

        flow = Flow([step_a], engine=MyCustomEngine())
        plan = flow.explain()
        assert plan.engine == "MyCustomEngine"

    def test_explain_empty_flow(self):
        """Empty flow produces valid plan with no nodes/edges."""
        flow = Flow([])
        plan = flow.explain()
        assert plan.original_nodes == []
        assert plan.original_edges == []
        assert plan.optimized_nodes == []
        assert plan.optimized_edges == []

    def test_explain_isolated_nodes_no_edges(self):
        """Dict with no edges produces plan with nodes but empty edges."""
        flow = Flow({"x": step_a, "y": step_b})
        plan = flow.explain()
        assert len(plan.original_nodes) == 2
        assert plan.original_edges == []


# ---------------------------------------------------------------------------
# FlowPlan.__str__ produces human-readable output
# ---------------------------------------------------------------------------


class TestFlowPlanStr:
    """FlowPlan.__str__ produces a human-readable representation."""

    def test_str_contains_engine(self):
        """String output shows the engine name."""
        plan = FlowPlan(
            original_nodes=["a", "b"],
            original_edges=[("a", "b")],
            optimized_nodes=["a", "b"],
            optimized_edges=[("a", "b")],
            engine="sequential",
        )
        output = str(plan)
        assert "Engine: sequential" in output

    def test_str_contains_nodes(self):
        """String output lists the node names."""
        plan = FlowPlan(
            original_nodes=["alpha", "beta"],
            original_edges=[("alpha", "beta")],
            optimized_nodes=["alpha", "beta"],
            optimized_edges=[("alpha", "beta")],
            engine="auto",
        )
        output = str(plan)
        assert "alpha" in output
        assert "beta" in output

    def test_str_contains_edges(self):
        """String output shows edge relationships."""
        plan = FlowPlan(
            original_nodes=["a", "b"],
            original_edges=[("a", "b")],
            optimized_nodes=["a", "b"],
            optimized_edges=[("a", "b")],
            engine="auto",
        )
        output = str(plan)
        assert "a -> b" in output

    def test_str_no_edges(self):
        """When there are no edges, output shows '(none)'."""
        plan = FlowPlan(
            original_nodes=["x"],
            original_edges=[],
            optimized_nodes=["x"],
            optimized_edges=[],
            engine="auto",
        )
        output = str(plan)
        assert "(none)" in output

    def test_str_no_applied_rules(self):
        """When no rules applied, output says '(none)'."""
        plan = FlowPlan(
            original_nodes=["a"],
            original_edges=[],
            optimized_nodes=["a"],
            optimized_edges=[],
            engine="auto",
            applied_rules=[],
        )
        output = str(plan)
        assert "Applied rules: (none)" in output

    def test_str_with_applied_rules(self):
        """Applied rules are listed."""
        plan = FlowPlan(
            original_nodes=["a", "b"],
            original_edges=[("a", "b")],
            optimized_nodes=["a", "b"],
            optimized_edges=[("a", "b")],
            engine="parallel",
            applied_rules=["parallelize", "dead_node_elimination"],
        )
        output = str(plan)
        assert "parallelize" in output
        assert "dead_node_elimination" in output

    def test_str_shows_optimized_when_different(self):
        """When optimized differs from original, both are shown."""
        plan = FlowPlan(
            original_nodes=["a", "b", "c"],
            original_edges=[("a", "b"), ("b", "c")],
            optimized_nodes=["a", "c"],
            optimized_edges=[("a", "c")],
            engine="sequential",
            applied_rules=["dead_node_elimination"],
        )
        output = str(plan)
        assert "Original graph:" in output
        assert "Optimized graph:" in output

    def test_str_header(self):
        """Output starts with a header."""
        plan = FlowPlan(
            original_nodes=["a"],
            original_edges=[],
            optimized_nodes=["a"],
            optimized_edges=[],
            engine="auto",
        )
        output = str(plan)
        assert "=== Flow Plan ===" in output


# ---------------------------------------------------------------------------
# Flow satisfies isinstance(flow, Runnable) check (Req 6.7)
# ---------------------------------------------------------------------------


class TestFlowIsRunnable:
    """Flow satisfies the Runnable protocol check."""

    def test_flow_is_instance_of_runnable(self):
        """Flow passes isinstance check against the runtime-checkable Runnable."""
        flow = Flow([step_a, step_b])
        assert isinstance(flow, Runnable)

    def test_flow_has_arun_method(self):
        """Flow has an arun method (the Runnable contract)."""
        flow = Flow([step_a])
        assert hasattr(flow, "arun")
        assert callable(flow.arun)

    def test_single_node_flow_is_runnable(self):
        """Even a single-node flow is a Runnable."""
        flow = Flow([step_a])
        assert isinstance(flow, Runnable)

    def test_empty_flow_is_runnable(self):
        """Even an empty flow is a Runnable."""
        flow = Flow([])
        assert isinstance(flow, Runnable)

    def test_dict_flow_is_runnable(self):
        """Dict-constructed flow is a Runnable."""
        flow = Flow({"a": step_a, "b": step_b}, edges=[Edge(source="a", target="b")])
        assert isinstance(flow, Runnable)
