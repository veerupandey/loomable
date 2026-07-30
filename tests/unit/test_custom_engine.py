"""Tests for custom ExecutionEngine protocol conformance.

Confirms that a developer-supplied custom engine satisfying the ExecutionEngine
protocol runs a Flow unchanged (Req 8.5).

Validates: Requirements 8.5
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.base import ExecutionEngine
from loomable.flow.flow import Flow, FlowPlan
from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create an AgentOutput with a single text part."""
    return AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )


class CustomEngine:
    """A custom engine satisfying the ExecutionEngine protocol.

    Records whether it was called and returns a predetermined RunResult.
    """

    def __init__(self) -> None:
        self.called = False
        self.received_flow = None
        self.received_input = None
        self.received_state = None
        self.received_context = None

    async def run(
        self,
        flow: "Flow",
        input,  # noqa: A002
        state: "SharedState",
        context: "RunContext",
    ) -> "RunResult":
        self.called = True
        self.received_flow = flow
        self.received_input = input
        self.received_state = state
        self.received_context = context

        output = _make_output(f"custom_result:{input}")
        return RunResult(output=output, session_id="custom-session")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCustomEngineSatisfiesProtocol:
    """Confirm a custom engine satisfying the protocol is recognized."""

    def test_custom_engine_satisfies_protocol(self):
        """A custom engine instance isinstance-checks as ExecutionEngine."""
        engine = CustomEngine()
        assert isinstance(engine, ExecutionEngine)


class TestCustomEngineRunsFlowUnchanged:
    """Confirm a custom engine runs a Flow without change to the Flow definition."""

    @pytest.mark.asyncio
    async def test_custom_engine_run_method_called(self):
        """WHEN a custom engine is passed to Flow, THEN its run() is called."""
        custom_engine = CustomEngine()

        async def step_a(input):
            return f"a:{input}"

        async def step_b(input):
            return f"b:{input}"

        flow = Flow({"a": step_a, "b": step_b}, engine=custom_engine)
        await flow.arun("hello")

        assert custom_engine.called is True

    @pytest.mark.asyncio
    async def test_custom_engine_receives_flow_reference(self):
        """The custom engine receives the Flow instance as first argument."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        await flow.arun("test")

        assert custom_engine.received_flow is flow

    @pytest.mark.asyncio
    async def test_custom_engine_receives_input(self):
        """The custom engine receives the input passed to flow.arun()."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        await flow.arun("my_input")

        assert custom_engine.received_input == "my_input"

    @pytest.mark.asyncio
    async def test_custom_engine_receives_shared_state(self):
        """The custom engine receives a SharedState instance."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        await flow.arun("test")

        assert isinstance(custom_engine.received_state, SharedState)

    @pytest.mark.asyncio
    async def test_custom_engine_receives_run_context(self):
        """The custom engine receives a RunContext instance."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        await flow.arun("test")

        assert isinstance(custom_engine.received_context, RunContext)

    @pytest.mark.asyncio
    async def test_custom_engine_result_returned(self):
        """The RunResult from the custom engine is what flow.arun() returns."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        result = await flow.arun("world")

        # The custom engine returns "custom_result:<input>"
        assert result.session_id == "custom-session"
        # Verify the output content
        output_data = result.output.parts[0].data.decode()
        assert output_data == "custom_result:world"

    @pytest.mark.asyncio
    async def test_flow_plan_records_custom_engine_class_name(self):
        """FlowPlan.engine records the custom engine's class name."""
        custom_engine = CustomEngine()

        async def step_a(input):
            return f"a:{input}"

        flow = Flow([step_a], engine=custom_engine)
        result = await flow.arun("test")

        plan = result.metadata["flow_plan"]
        assert isinstance(plan, FlowPlan)
        assert plan.engine == "CustomEngine"

    @pytest.mark.asyncio
    async def test_flow_definition_unchanged_with_custom_engine(self):
        """The Flow's node/edge structure is unchanged regardless of engine choice."""
        custom_engine = CustomEngine()

        async def step_a(input):
            return f"a:{input}"

        async def step_b(input):
            return f"b:{input}"

        # Create flow with custom engine
        flow = Flow({"a": step_a, "b": step_b}, engine=custom_engine)

        # The flow's internal structure should be intact
        assert "a" in flow.nodes
        assert "b" in flow.nodes

        # explain() should reflect the custom engine name
        plan = flow.explain()
        assert plan.engine == "CustomEngine"
        assert sorted(plan.original_nodes) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_custom_engine_with_deps(self):
        """A custom engine receives context with deps when configured on the Flow."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        deps = {"db": "fake_connection"}
        flow = Flow([step], engine=custom_engine, deps=deps)
        await flow.arun("test")

        assert custom_engine.received_context.deps == deps

    @pytest.mark.asyncio
    async def test_resolve_engine_returns_custom_object_directly(self):
        """Flow._resolve_engine returns the custom engine object directly."""
        custom_engine = CustomEngine()

        async def step(input):
            return f"done:{input}"

        flow = Flow([step], engine=custom_engine)
        resolved = flow._resolve_engine()

        assert resolved is custom_engine
