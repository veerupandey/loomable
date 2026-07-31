# Feature: workflow-ergonomics, Property 7: Workflow compiles sequential steps into ordered edges
"""Property 7: Workflow compiles sequential steps into ordered edges.

For any Workflow containing N sequential Steps, the compiled Flow SHALL contain
N-1 edges connecting them in declaration order (step[i] → step[i+1] for all
i in 0..N-2).

**Validates: Requirements 4.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate distinct step names (2-8 steps for meaningful edge checks)
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
    max_size=8,
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
# Property tests
# ---------------------------------------------------------------------------


class TestWorkflowEdgeCompilation:
    """Property 7: Workflow compiles sequential steps into ordered edges."""

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_n_steps_produce_n_minus_1_edges(self, names: list[str]) -> None:
        """For N sequential steps, the compiled Flow has exactly N-1 edges
        connecting them in declaration order."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: exactly N-1 edges for N sequential steps
        n = len(names)
        assert len(plan.original_edges) == n - 1, (
            f"Expected {n - 1} edges for {n} steps, got {len(plan.original_edges)}"
        )

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_edges_connect_steps_in_declaration_order(self, names: list[str]) -> None:
        """Each edge connects step[i] to step[i+1] in declaration order."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: edge i connects names[i] → names[i+1]
        for i in range(len(names) - 1):
            expected_edge = (names[i], names[i + 1])
            assert expected_edge in plan.original_edges, (
                f"Expected edge {expected_edge} not found in {plan.original_edges}"
            )

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_all_step_names_appear_in_nodes(self, names: list[str]) -> None:
        """All step names appear in the FlowPlan's original_nodes."""
        steps = [_make_step(name) for name in names]
        wf = Workflow(name="test_workflow", steps=steps)

        plan = wf.explain()

        # Property: every step name is in original_nodes
        for name in names:
            assert name in plan.original_nodes, (
                f"Step name '{name}' not found in original_nodes: {plan.original_nodes}"
            )
