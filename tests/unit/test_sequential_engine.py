"""Tests for SequentialEngine and Flow.arun end-to-end wiring.

Covers:
- Sequential run produces correct ordered state (each node sees previous outputs)
- Edge conditions are honored (node skipped when condition is False)
- Flow.arun works end-to-end with a simple list shorthand
- Flow.arun works end-to-end with dict + edges
- sub_results contains per-node results
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.sequential import SequentialEngine
from loomable.flow.flow import Flow, FlowPlan
from loomable.flow.nodes import Edge, Node
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create a simple text AgentOutput."""
    return AgentOutput(
        parts=[
            MediaPart(
                modality=Modality.TEXT,
                media_type="text/plain",
                data=text.encode("utf-8"),
            )
        ]
    )


def _output_text(output: AgentOutput) -> str:
    """Extract text from an AgentOutput."""
    return output.parts[0].data.decode("utf-8")


# ---------------------------------------------------------------------------
# SequentialEngine unit tests
# ---------------------------------------------------------------------------


class TestSequentialEngineOrdering:
    """Test that the sequential engine runs nodes in topological order."""

    @pytest.mark.asyncio
    async def test_linear_chain_produces_correct_state(self):
        """A→B→C: each node appends to the input, state reflects order."""
        execution_order = []

        def step_a(inp):
            execution_order.append("A")
            return f"{inp}->A"

        def step_b(inp):
            execution_order.append("B")
            # inp here should be the AgentOutput from A
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            return f"{text}->B"

        def step_c(inp):
            execution_order.append("C")
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            return f"{text}->C"

        flow = Flow([step_a, step_b, step_c])
        result = await flow.arun("start")

        # Check execution order
        assert execution_order == ["A", "B", "C"]

        # Check final output is from the last node
        final_text = _output_text(result.output)
        assert "->C" in final_text

        # Check sub_results contains all nodes
        assert result.sub_results is not None
        assert len(result.sub_results) == 3

    @pytest.mark.asyncio
    async def test_diamond_dag_respects_dependencies(self):
        """Diamond: A→B, A→C, B→D, C→D. A runs first, B/C before D."""
        execution_order = []

        def step_a(inp):
            execution_order.append("A")
            return "A_done"

        def step_b(inp):
            execution_order.append("B")
            return "B_done"

        def step_c(inp):
            execution_order.append("C")
            return "C_done"

        def step_d(inp):
            execution_order.append("D")
            return "D_done"

        flow = Flow(
            nodes={
                "A": step_a,
                "B": step_b,
                "C": step_c,
                "D": step_d,
            },
            edges=[
                Edge(source="A", target="B"),
                Edge(source="A", target="C"),
                Edge(source="B", target="D"),
                Edge(source="C", target="D"),
            ],
        )

        result = await flow.arun("start")

        # A must run before B and C; B and C before D
        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("A") < execution_order.index("C")
        assert execution_order.index("B") < execution_order.index("D")
        assert execution_order.index("C") < execution_order.index("D")


class TestSequentialEngineEdgeConditions:
    """Test that edge conditions are honored (nodes skipped when falsy)."""

    @pytest.mark.asyncio
    async def test_node_skipped_when_all_conditions_false(self):
        """When all incoming edges have conditions that are False, node is skipped."""
        executed = []

        def step_a(inp):
            executed.append("A")
            return "A_done"

        def step_b(inp):
            executed.append("B")
            return "B_done"

        def step_c(inp):
            executed.append("C")
            return "C_done"

        # A → B (condition: always False) → C (unconditional from B)
        # B should be skipped because its only incoming condition is False
        flow = Flow(
            nodes={"A": step_a, "B": step_b, "C": step_c},
            edges=[
                Edge(source="A", target="B", condition=lambda state: False),
                Edge(source="B", target="C"),
            ],
        )

        result = await flow.arun("start")

        # B should be skipped (all incoming edges are conditional and all false)
        assert "A" in executed
        assert "B" not in executed
        # C has an unconditional incoming edge from B, so it executes
        assert "C" in executed

    @pytest.mark.asyncio
    async def test_node_executes_when_condition_true(self):
        """When an incoming edge condition evaluates True, node executes."""
        executed = []

        def step_a(inp):
            executed.append("A")
            return "A_done"

        def step_b(inp):
            executed.append("B")
            return "B_done"

        flow = Flow(
            nodes={"A": step_a, "B": step_b},
            edges=[
                Edge(source="A", target="B", condition=lambda state: True),
            ],
        )

        result = await flow.arun("start")

        assert executed == ["A", "B"]

    @pytest.mark.asyncio
    async def test_conditional_routing_based_on_state(self):
        """Edge condition reads from state to decide routing."""
        executed = []

        def step_a(inp):
            executed.append("A")
            return "go_left"

        def step_left(inp):
            executed.append("left")
            return "left_done"

        def step_right(inp):
            executed.append("right")
            return "right_done"

        # A → left (condition: state["A"] text contains "go_left")
        # A → right (condition: state["A"] text contains "go_right")
        flow = Flow(
            nodes={"A": step_a, "left": step_left, "right": step_right},
            edges=[
                Edge(
                    source="A",
                    target="left",
                    condition=lambda state: (
                        state.get("A") is not None
                        and "go_left" in _output_text(state.get("A"))
                    ),
                ),
                Edge(
                    source="A",
                    target="right",
                    condition=lambda state: (
                        state.get("A") is not None
                        and "go_right" in _output_text(state.get("A"))
                    ),
                ),
            ],
        )

        result = await flow.arun("start")

        assert "A" in executed
        assert "left" in executed
        assert "right" not in executed


class TestFlowArunEndToEnd:
    """Test Flow.arun works end-to-end."""

    @pytest.mark.asyncio
    async def test_list_shorthand_end_to_end(self):
        """Flow with list shorthand runs all nodes sequentially."""

        def double(inp):
            val = int(inp) if isinstance(inp, str) else int(
                _output_text(inp) if isinstance(inp, AgentOutput) else inp
            )
            return str(val * 2)

        def add_one(inp):
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            val = int(text)
            return str(val + 1)

        flow = Flow([double, add_one])
        result = await flow.arun("5")

        # 5 * 2 = 10, 10 + 1 = 11
        final_text = _output_text(result.output)
        assert final_text == "11"

    @pytest.mark.asyncio
    async def test_dict_with_edges_end_to_end(self):
        """Flow with explicit dict nodes and edges runs correctly."""

        def greet(inp):
            return f"Hello, {inp}"

        def shout(inp):
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            return text.upper()

        flow = Flow(
            nodes={"greet": greet, "shout": shout},
            edges=[Edge(source="greet", target="shout")],
        )

        result = await flow.arun("World")

        final_text = _output_text(result.output)
        assert final_text == "HELLO, WORLD"

    @pytest.mark.asyncio
    async def test_single_node_flow(self):
        """A flow with a single node works."""

        def echo(inp):
            return f"echo: {inp}"

        flow = Flow([echo])
        result = await flow.arun("test")

        final_text = _output_text(result.output)
        assert final_text == "echo: test"

    @pytest.mark.asyncio
    async def test_flow_plan_attached_to_result(self):
        """Flow.arun attaches a FlowPlan to result metadata."""

        def noop(inp):
            return "done"

        flow = Flow([noop])
        result = await flow.arun("x")

        assert "flow_plan" in result.metadata
        plan = result.metadata["flow_plan"]
        assert isinstance(plan, FlowPlan)
        assert plan.engine == "SequentialEngine"

    @pytest.mark.asyncio
    async def test_context_passes_through(self):
        """RunContext is passed through to nodes."""
        received_deps = []

        def capture_deps(inp, *, context=None):
            if context is not None:
                received_deps.append(context.deps)
            return "done"

        flow = Flow([capture_deps], deps={"api_key": "secret"})
        ctx = RunContext()
        result = await flow.arun("x", context=ctx)

        assert received_deps == [{"api_key": "secret"}]


class TestSubResults:
    """Test that sub_results contains per-node results."""

    @pytest.mark.asyncio
    async def test_sub_results_keyed_by_node_id(self):
        """sub_results maps node_id -> RunResult for each executed node."""

        def step_a(inp):
            return "result_a"

        def step_b(inp):
            return "result_b"

        flow = Flow(
            nodes={"A": step_a, "B": step_b},
            edges=[Edge(source="A", target="B")],
        )

        result = await flow.arun("start")

        assert result.sub_results is not None
        assert "A" in result.sub_results
        assert "B" in result.sub_results

        # Verify each sub_result is a RunResult with the node's output
        a_text = _output_text(result.sub_results["A"].output)
        b_text = _output_text(result.sub_results["B"].output)
        assert a_text == "result_a"
        assert b_text == "result_b"

    @pytest.mark.asyncio
    async def test_skipped_nodes_not_in_sub_results(self):
        """Nodes that were skipped (condition False) are not in sub_results."""

        def step_a(inp):
            return "done_a"

        def step_b(inp):
            return "done_b"

        flow = Flow(
            nodes={"A": step_a, "B": step_b},
            edges=[
                Edge(source="A", target="B", condition=lambda state: False),
            ],
        )

        result = await flow.arun("start")

        assert result.sub_results is not None
        assert "A" in result.sub_results
        assert "B" not in result.sub_results

    @pytest.mark.asyncio
    async def test_state_propagation_between_nodes(self):
        """Each node can see previous nodes' outputs via state (Req 7.1)."""
        seen_inputs = []

        def step_a(inp):
            seen_inputs.append(("A", inp))
            return "from_A"

        def step_b(inp):
            # B should receive A's output (AgentOutput) as input
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            seen_inputs.append(("B", text))
            return f"from_B(got:{text})"

        def step_c(inp):
            text = _output_text(inp) if isinstance(inp, AgentOutput) else str(inp)
            seen_inputs.append(("C", text))
            return f"from_C(got:{text})"

        flow = Flow(
            nodes={"A": step_a, "B": step_b, "C": step_c},
            edges=[
                Edge(source="A", target="B"),
                Edge(source="B", target="C"),
            ],
        )

        result = await flow.arun("initial")

        # A gets the initial input
        assert seen_inputs[0] == ("A", "initial")
        # B gets A's output
        assert seen_inputs[1] == ("B", "from_A")
        # C gets B's output
        assert seen_inputs[2] == ("C", "from_B(got:from_A)")
