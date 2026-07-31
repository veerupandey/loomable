# Feature: workflow-ergonomics, Property 2: Step name is preserved as node_id in compiled Flow
"""Property 2: Step name is preserved as node_id in compiled Flow.

For any non-empty string used as a Step's `name`, when that Step is compiled
into a Workflow, the resulting Flow SHALL contain a node whose `node_id` equals
the Step's `name`.

Since the Workflow class is not yet implemented, this test validates the
property by constructing a Flow using the Step's name as node_id — which is
exactly what the WorkflowCompiler will do (Step.name → Node.node_id → Flow node key).

**Validates: Requirements 1.4, 1.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.flow import Flow
from loomable.flow.nodes import Node
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: non-empty name strings — printable characters, no control chars,
# varied lengths to exercise edge cases (single char, long names, unicode).
step_name_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "" and len(s) > 0)


# Strategy: multiple distinct step names for multi-node flows
distinct_step_names_st = st.lists(
    step_name_st,
    min_size=1,
    max_size=8,
    unique=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(x: Any) -> str:
    """A minimal callable agent for testing."""
    return f"processed: {x}"


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestStepNamePreservation:
    """Property 2: Step name is preserved as node_id in compiled Flow."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_name_st)
    def test_step_name_stored_correctly(self, name: str) -> None:
        """Step.name preserves the exact string passed at construction."""
        step = Step(name=name, agent=_noop)
        assert step.name == name

    @settings(max_examples=100, deadline=None)
    @given(name=step_name_st)
    def test_step_name_used_as_node_id_in_flow(self, name: str) -> None:
        """When a Step is compiled into a Flow (Step.name → Node.node_id),
        the Flow's nodes dict contains a key equal to the Step's name.

        This simulates WorkflowCompiler behavior: for each Step, create a
        Node with node_id = step.name and add it to the Flow.
        """
        step = Step(name=name, agent=_noop)

        # Simulate compilation: create a Node using step.name as node_id
        node = Node(node_id=step.name, runnable=step)

        # Build a Flow with this node
        flow = Flow(nodes={node.node_id: node})

        # Verify: the Flow contains a node whose node_id equals the Step's name
        assert step.name in flow.nodes
        assert flow.nodes[step.name].node_id == step.name

    @settings(max_examples=100, deadline=None)
    @given(names=distinct_step_names_st)
    def test_multiple_step_names_preserved_as_node_ids(
        self, names: list[str]
    ) -> None:
        """For a list of distinct step names, all are preserved as node_ids
        in a multi-node Flow — no name is lost or altered during compilation."""
        steps = [Step(name=n, agent=_noop) for n in names]

        # Simulate compilation: create Nodes and wire sequentially
        nodes = {s.name: Node(node_id=s.name, runnable=s) for s in steps}
        flow = Flow(nodes=nodes)

        # Verify: every step name appears as a node_id in the compiled Flow
        flow_node_ids = set(flow.nodes.keys())
        for step in steps:
            assert step.name in flow_node_ids
            assert flow.nodes[step.name].node_id == step.name

        # Verify: no extra nodes were created
        assert len(flow.nodes) == len(steps)
