# Feature: workflow-ergonomics, Property 14: explain() returns topology with step names pre-execution
"""Property 14: explain() returns topology with step names pre-execution.

For any Workflow or FlowClass, calling explain() before any execution SHALL
return a FlowPlan whose original_nodes list contains the Step names (or method
names for FlowClass) used during construction.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.flow import FlowPlan
from loomable.flow.flow_class import FlowClass, listen, start
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate distinct step names (2-6 steps as spec indicates)
step_name_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")

distinct_step_names_st = st.lists(
    step_name_st,
    min_size=2,
    max_size=6,
    unique=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(x: Any) -> str:
    """A minimal callable agent for testing."""
    return f"processed: {x}"


def _make_step(name: str) -> Step:
    """Create a Step with a no-op callable agent."""
    return Step(name=name, agent=_noop)


# ---------------------------------------------------------------------------
# Fixed FlowClass topologies for testing
# ---------------------------------------------------------------------------


class LinearFlowClass(FlowClass):
    """A simple linear FlowClass: begin -> process -> finalize."""

    @start()
    def begin(self, input: Any) -> str:
        return f"started: {input}"

    @listen("begin")
    def process(self, input: Any) -> str:
        return f"processed: {input}"

    @listen("process")
    def finalize(self, input: Any) -> str:
        return f"finalized: {input}"


class FanOutFlowClass(FlowClass):
    """A fan-out FlowClass: entry -> analyze, entry -> summarize."""

    @start()
    def entry(self, input: Any) -> str:
        return f"entry: {input}"

    @listen("entry")
    def analyze(self, input: Any) -> str:
        return f"analyzed: {input}"

    @listen("entry")
    def summarize(self, input: Any) -> str:
        return f"summarized: {input}"


class SingleNodeFlowClass(FlowClass):
    """Minimal FlowClass with just one start method."""

    @start()
    def run_task(self, input: Any) -> str:
        return f"ran: {input}"


class DiamondFlowClass(FlowClass):
    """Diamond topology: start -> left, start -> right, left -> merge, right -> merge."""

    @start()
    def begin(self, input: Any) -> str:
        return f"begin: {input}"

    @listen("begin")
    def left_branch(self, input: Any) -> str:
        return f"left: {input}"

    @listen("begin")
    def right_branch(self, input: Any) -> str:
        return f"right: {input}"

    @listen("left_branch")
    def merge(self, input: Any) -> str:
        return f"merged: {input}"


# ---------------------------------------------------------------------------
# Property tests — Workflow
# ---------------------------------------------------------------------------


class TestExplainPreExecutionWorkflow:
    """Property 14: explain() returns topology with step names pre-execution (Workflow)."""

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_explain_returns_flowplan_without_execution(self, names: list[str]) -> None:
        """Calling explain() before any execution returns a FlowPlan."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        # Call explain() WITHOUT executing arun/run first
        plan = wf.explain()

        # Property: explain() returns a FlowPlan instance
        assert isinstance(plan, FlowPlan), (
            f"Expected FlowPlan, got {type(plan).__name__}"
        )

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_original_nodes_contains_all_step_names(self, names: list[str]) -> None:
        """FlowPlan.original_nodes contains all Step names used during construction."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: every step name appears in original_nodes
        for name in names:
            assert name in plan.original_nodes, (
                f"Step name '{name}' not found in original_nodes: {plan.original_nodes}"
            )

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_original_edges_shows_sequential_connections(self, names: list[str]) -> None:
        """FlowPlan.original_edges shows the sequential connections between steps."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: sequential steps are connected in declaration order
        for i in range(len(names) - 1):
            expected_edge = (names[i], names[i + 1])
            assert expected_edge in plan.original_edges, (
                f"Expected sequential edge {expected_edge} not found in "
                f"original_edges: {plan.original_edges}"
            )

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_explain_uses_step_names_as_node_ids(self, names: list[str]) -> None:
        """The FlowPlan uses Step names (not internal IDs) as node identifiers."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: the original_nodes list has exactly the step names
        # (there may be additional internal nodes for conditions, but for
        # pure sequential steps, it should be exactly the step names)
        assert set(names) == set(plan.original_nodes), (
            f"Expected nodes {set(names)}, got {set(plan.original_nodes)}"
        )


# ---------------------------------------------------------------------------
# Property tests — FlowClass
# ---------------------------------------------------------------------------


class TestExplainPreExecutionFlowClass:
    """Property 14: explain() returns topology with step names pre-execution (FlowClass)."""

    def test_linear_flowclass_explain_pre_execution(self) -> None:
        """LinearFlowClass.explain() returns FlowPlan with method names before execution."""
        flow = LinearFlowClass()

        plan = flow.explain()

        assert isinstance(plan, FlowPlan)
        expected_methods = {"begin", "process", "finalize"}
        assert expected_methods.issubset(set(plan.original_nodes)), (
            f"Expected methods {expected_methods} in original_nodes: {plan.original_nodes}"
        )

    def test_linear_flowclass_edges_match_listen_topology(self) -> None:
        """LinearFlowClass edges reflect @listen decorator connections."""
        flow = LinearFlowClass()

        plan = flow.explain()

        # begin -> process, process -> finalize
        assert ("begin", "process") in plan.original_edges
        assert ("process", "finalize") in plan.original_edges

    def test_fanout_flowclass_explain_pre_execution(self) -> None:
        """FanOutFlowClass.explain() returns FlowPlan with all method names."""
        flow = FanOutFlowClass()

        plan = flow.explain()

        assert isinstance(plan, FlowPlan)
        expected_methods = {"entry", "analyze", "summarize"}
        assert expected_methods.issubset(set(plan.original_nodes)), (
            f"Expected methods {expected_methods} in original_nodes: {plan.original_nodes}"
        )

    def test_fanout_flowclass_edges_show_fanout(self) -> None:
        """FanOutFlowClass edges show fan-out from entry to both listeners."""
        flow = FanOutFlowClass()

        plan = flow.explain()

        # entry -> analyze and entry -> summarize
        assert ("entry", "analyze") in plan.original_edges
        assert ("entry", "summarize") in plan.original_edges

    def test_single_node_flowclass_explain_pre_execution(self) -> None:
        """SingleNodeFlowClass.explain() returns FlowPlan with the start method."""
        flow = SingleNodeFlowClass()

        plan = flow.explain()

        assert isinstance(plan, FlowPlan)
        assert "run_task" in plan.original_nodes

    def test_diamond_flowclass_explain_pre_execution(self) -> None:
        """DiamondFlowClass.explain() returns FlowPlan with all method names."""
        flow = DiamondFlowClass()

        plan = flow.explain()

        assert isinstance(plan, FlowPlan)
        expected_methods = {"begin", "left_branch", "right_branch", "merge"}
        assert expected_methods.issubset(set(plan.original_nodes)), (
            f"Expected methods {expected_methods} in original_nodes: {plan.original_nodes}"
        )

    def test_diamond_flowclass_edges_match_topology(self) -> None:
        """DiamondFlowClass edges reflect the diamond listener topology."""
        flow = DiamondFlowClass()

        plan = flow.explain()

        # begin -> left_branch, begin -> right_branch, left_branch -> merge
        assert ("begin", "left_branch") in plan.original_edges
        assert ("begin", "right_branch") in plan.original_edges
        assert ("left_branch", "merge") in plan.original_edges

    def test_flowclass_explain_available_before_execution(self) -> None:
        """All FlowClass instances can call explain() without any prior execution."""
        for cls in [LinearFlowClass, FanOutFlowClass, SingleNodeFlowClass, DiamondFlowClass]:
            flow = cls()
            plan = flow.explain()
            # Just calling explain() without executing should not raise
            assert isinstance(plan, FlowPlan), (
                f"{cls.__name__}.explain() did not return a FlowPlan"
            )
            # original_nodes should not be empty
            assert len(plan.original_nodes) > 0, (
                f"{cls.__name__}.explain() returned empty original_nodes"
            )
