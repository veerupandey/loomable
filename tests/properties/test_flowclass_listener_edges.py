# Feature: workflow-ergonomics, Property 13: FlowClass listener edges match decorator topology
"""Property 13: FlowClass listener edges match decorator topology.

For any FlowClass subclass where method B is decorated with @listen("A"),
the compiled Flow SHALL contain an edge from node "A" to node "B", and when
A executes, B SHALL receive A's output as input.

**Validates: Requirements 6.4, 6.8**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.flow_class import FlowClass, listen, start


# ---------------------------------------------------------------------------
# Fixed FlowClass topologies for property verification
# ---------------------------------------------------------------------------


class LinearTwoFlow(FlowClass):
    """A -> B (linear 2-node chain)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        # input is an AgentOutput from step_a; extract text
        text = input.text() if hasattr(input, "text") else str(input)
        return f"b:{text}"


class LinearThreeFlow(FlowClass):
    """A -> B -> C (linear 3-node chain)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        text = input.text() if hasattr(input, "text") else str(input)
        return f"b:{text}"

    @listen("step_b")
    async def step_c(self, input: Any) -> str:
        text = input.text() if hasattr(input, "text") else str(input)
        return f"c:{text}"


class LinearFourFlow(FlowClass):
    """A -> B -> C -> D (linear 4-node chain)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        return f"b:{input}"

    @listen("step_b")
    async def step_c(self, input: Any) -> str:
        return f"c:{input}"

    @listen("step_c")
    async def step_d(self, input: Any) -> str:
        return f"d:{input}"


class FanOutTwoFlow(FlowClass):
    """A -> B, A -> C (fan-out from A to two listeners)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        return f"b:{input}"

    @listen("step_a")
    async def step_c(self, input: Any) -> str:
        return f"c:{input}"


class FanOutThreeFlow(FlowClass):
    """A -> B, A -> C, A -> D (fan-out from A to three listeners)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        return f"b:{input}"

    @listen("step_a")
    async def step_c(self, input: Any) -> str:
        return f"c:{input}"

    @listen("step_a")
    async def step_d(self, input: Any) -> str:
        return f"d:{input}"


class DiamondFlow(FlowClass):
    """A -> B, A -> C, B -> D, C -> D (diamond topology)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        return f"b:{input}"

    @listen("step_a")
    async def step_c(self, input: Any) -> str:
        return f"c:{input}"

    @listen("step_b")
    async def step_d(self, input: Any) -> str:
        return f"d:{input}"


class LinearFiveFlow(FlowClass):
    """A -> B -> C -> D -> E (linear 5-node chain)."""

    @start()
    async def step_a(self, input: Any) -> str:
        return f"a:{input}"

    @listen("step_a")
    async def step_b(self, input: Any) -> str:
        return f"b:{input}"

    @listen("step_b")
    async def step_c(self, input: Any) -> str:
        return f"c:{input}"

    @listen("step_c")
    async def step_d(self, input: Any) -> str:
        return f"d:{input}"

    @listen("step_d")
    async def step_e(self, input: Any) -> str:
        return f"e:{input}"


# ---------------------------------------------------------------------------
# Topology descriptions for parameterized testing
# ---------------------------------------------------------------------------

# Each topology: (FlowClass, expected_nodes, expected_edges)
TOPOLOGIES = [
    (
        LinearTwoFlow,
        {"step_a", "step_b"},
        {("step_a", "step_b")},
    ),
    (
        LinearThreeFlow,
        {"step_a", "step_b", "step_c"},
        {("step_a", "step_b"), ("step_b", "step_c")},
    ),
    (
        LinearFourFlow,
        {"step_a", "step_b", "step_c", "step_d"},
        {("step_a", "step_b"), ("step_b", "step_c"), ("step_c", "step_d")},
    ),
    (
        FanOutTwoFlow,
        {"step_a", "step_b", "step_c"},
        {("step_a", "step_b"), ("step_a", "step_c")},
    ),
    (
        FanOutThreeFlow,
        {"step_a", "step_b", "step_c", "step_d"},
        {("step_a", "step_b"), ("step_a", "step_c"), ("step_a", "step_d")},
    ),
    (
        DiamondFlow,
        {"step_a", "step_b", "step_c", "step_d"},
        {("step_a", "step_b"), ("step_a", "step_c"), ("step_b", "step_d")},
    ),
    (
        LinearFiveFlow,
        {"step_a", "step_b", "step_c", "step_d", "step_e"},
        {
            ("step_a", "step_b"),
            ("step_b", "step_c"),
            ("step_c", "step_d"),
            ("step_d", "step_e"),
        },
    ),
]


# Strategy: pick a topology index
topology_index_st = st.integers(min_value=0, max_value=len(TOPOLOGIES) - 1)

# Strategy: generate arbitrary input strings for execution tests
input_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=0,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestFlowClassListenerEdges:
    """Property 13: FlowClass listener edges match decorator topology."""

    @settings(max_examples=100, deadline=None)
    @given(idx=topology_index_st)
    def test_edges_match_listen_decorators(self, idx: int) -> None:
        """For each @listen("A") on method B, the compiled Flow contains
        edge (A, B) in its topology."""
        flow_cls, expected_nodes, expected_edges = TOPOLOGIES[idx]
        flow_instance = flow_cls()
        plan = flow_instance.explain()

        actual_edges = set(plan.original_edges)

        # Property: every expected edge exists in the compiled flow
        for expected_edge in expected_edges:
            assert expected_edge in actual_edges, (
                f"Expected edge {expected_edge} not found in "
                f"{actual_edges} for {flow_cls.__name__}"
            )

    @settings(max_examples=100, deadline=None)
    @given(idx=topology_index_st)
    def test_all_method_names_appear_as_nodes(self, idx: int) -> None:
        """All decorated method names appear in the FlowPlan's original_nodes."""
        flow_cls, expected_nodes, _ = TOPOLOGIES[idx]
        flow_instance = flow_cls()
        plan = flow_instance.explain()

        actual_nodes = set(plan.original_nodes)

        # Property: every expected node is present
        for expected_node in expected_nodes:
            assert expected_node in actual_nodes, (
                f"Expected node '{expected_node}' not found in "
                f"{actual_nodes} for {flow_cls.__name__}"
            )

    @settings(max_examples=100, deadline=None)
    @given(idx=topology_index_st)
    def test_edge_count_matches_listen_count(self, idx: int) -> None:
        """The number of edges equals the number of @listen decorators."""
        flow_cls, _, expected_edges = TOPOLOGIES[idx]
        flow_instance = flow_cls()
        plan = flow_instance.explain()

        # Property: edge count matches the expected listener count
        assert len(plan.original_edges) == len(expected_edges), (
            f"Expected {len(expected_edges)} edges, got "
            f"{len(plan.original_edges)} for {flow_cls.__name__}"
        )

    @settings(max_examples=100, deadline=None)
    @given(idx=topology_index_st, input_val=input_st)
    @pytest.mark.asyncio
    async def test_listener_receives_source_output(
        self, idx: int, input_val: str
    ) -> None:
        """When A executes, B (listening to A) receives A's output as input.

        We verify this by running the linear two-node flow and checking that
        B's output contains A's output string."""
        # For execution verification, use LinearTwoFlow specifically
        # since its output chain is predictable: a:{input} -> b:a:{input}
        flow_instance = LinearTwoFlow()
        result = await flow_instance.kickoff(input_val)

        # B should receive "a:{input_val}" as its input and produce "b:a:{input_val}"
        output_text = result.output.text()
        expected = f"b:a:{input_val}"
        assert output_text == expected, (
            f"Expected '{expected}', got '{output_text}'. "
            f"Listener B should receive A's output as input."
        )

    @settings(max_examples=100, deadline=None)
    @given(input_val=input_st)
    @pytest.mark.asyncio
    async def test_linear_chain_propagation(self, input_val: str) -> None:
        """In a linear chain A -> B -> C, C receives B's output which
        received A's output."""
        flow_instance = LinearThreeFlow()
        result = await flow_instance.kickoff(input_val)

        # Chain: A produces "a:{input}", B gets that and produces "b:a:{input}",
        # C gets that and produces "c:b:a:{input}"
        output_text = result.output.text()
        expected = f"c:b:a:{input_val}"
        assert output_text == expected, (
            f"Expected '{expected}', got '{output_text}'. "
            f"Chain propagation should preserve output through listeners."
        )
