"""Unit tests for the Workflow class.

Validates:
- Req 4.1: Workflow accepts name, steps, deps, memory, session_id
- Req 4.2: arun delegates to compiled Flow
- Req 4.3: run() wraps arun in asyncio.run()
- Req 4.4: Workflow implements Runnable protocol
- Req 4.8: state property exposes SharedState post-execution
- Req 4.9: Empty steps raises ValueError
- Req 4.10: memory=True auto-creates TieredMemoryStore
- Req 7.1: explain() returns FlowPlan pre-execution
- Req 7.3: explain() available without running
- Req 10.1: Duplicate step names raise FlowConfigError
"""

import asyncio

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.flow import FlowPlan
from loomable.flow.nodes import FlowConfigError
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(name: str, output_text: str = "ok") -> Step:
    """Create a simple Step that returns a fixed text output."""

    async def handler(input, *, context=None):  # noqa: A002
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=output_text.encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")

    return Step(name=name, agent=handler)


def _make_state_writing_step(name: str, key: str, value: str) -> Step:
    """Create a Step that writes a key/value into SharedState."""

    async def handler(input, *, context=None):  # noqa: A002
        if context and context.shared_state:
            context.shared_state.write(key, value)
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=value.encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")

    return Step(name=name, agent=handler)


# ---------------------------------------------------------------------------
# Construction Validation Tests
# ---------------------------------------------------------------------------


class TestWorkflowConstruction:
    """Workflow construction validates inputs at creation time."""

    def test_empty_steps_raises_value_error(self):
        """Empty steps list raises ValueError."""
        with pytest.raises(ValueError, match="At least one step is required"):
            Workflow(name="test", steps=[])

    def test_duplicate_step_names_raises_flow_config_error(self):
        """Duplicate step names raise FlowConfigError."""
        step_a = _make_step("research")
        step_b = _make_step("research")

        with pytest.raises(FlowConfigError, match="Duplicate step name: 'research'"):
            Workflow(name="test", steps=[step_a, step_b])

    def test_valid_single_step_constructs_successfully(self):
        """A single step creates a valid Workflow."""
        step = _make_step("step_one")
        wf = Workflow(name="my_workflow", steps=[step])
        assert wf.name == "my_workflow"

    def test_valid_multiple_steps_constructs_successfully(self):
        """Multiple steps with unique names creates a valid Workflow."""
        steps = [_make_step("a"), _make_step("b"), _make_step("c")]
        wf = Workflow(name="pipeline", steps=steps)
        assert wf.name == "pipeline"

    def test_deps_stored(self):
        """Deps parameter is accepted."""
        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step], deps={"db": "connection"})
        assert wf.name == "test"


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestWorkflowExecution:
    """Workflow executes by delegating to the compiled Flow."""

    @pytest.mark.asyncio
    async def test_arun_returns_run_result(self):
        """arun returns a RunResult from the compiled Flow."""
        step = _make_step("greet", output_text="hello world")
        wf = Workflow(name="test", steps=[step])

        result = await wf.arun("input")
        assert isinstance(result, RunResult)
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_arun_sequential_pipeline(self):
        """Steps execute in declaration order, piping output through."""
        step_a = _make_step("first", output_text="from_first")
        step_b = _make_step("second", output_text="from_second")
        wf = Workflow(name="pipeline", steps=[step_a, step_b])

        result = await wf.arun("start")
        assert isinstance(result, RunResult)
        # The final output should come from the last step
        assert result.output.text() == "from_second"

    def test_run_sync_wrapper(self):
        """run() wraps arun in asyncio.run() for sync contexts."""
        step = _make_step("sync_test", output_text="sync_result")
        wf = Workflow(name="test", steps=[step])

        result = wf.run("input")
        assert isinstance(result, RunResult)
        assert result.output.text() == "sync_result"


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------


class TestWorkflowState:
    """Workflow.state exposes SharedState from the most recent run."""

    def test_state_before_execution_is_empty(self):
        """Before execution, state returns an empty SharedState."""
        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step])

        state = wf.state
        assert isinstance(state, SharedState)

    @pytest.mark.asyncio
    async def test_state_after_execution_reflects_writes(self):
        """After execution, state reflects what steps wrote."""
        step = _make_state_writing_step("writer", "result_key", "result_value")
        wf = Workflow(name="test", steps=[step])

        ctx = RunContext()
        await wf.arun("input", context=ctx)

        # The state should have the written key
        state = wf.state
        assert state.get("result_key") == "result_value"


# ---------------------------------------------------------------------------
# Explain Tests
# ---------------------------------------------------------------------------


class TestWorkflowExplain:
    """Workflow.explain() returns FlowPlan without requiring execution."""

    def test_explain_returns_flow_plan(self):
        """explain() returns a FlowPlan instance."""
        steps = [_make_step("a"), _make_step("b"), _make_step("c")]
        wf = Workflow(name="pipeline", steps=steps)

        plan = wf.explain()
        assert isinstance(plan, FlowPlan)

    def test_explain_contains_step_names(self):
        """FlowPlan's original_nodes contain the step names."""
        steps = [_make_step("research"), _make_step("analyze"), _make_step("report")]
        wf = Workflow(name="pipeline", steps=steps)

        plan = wf.explain()
        # Step names should be present in the original nodes
        assert "research" in plan.original_nodes
        assert "analyze" in plan.original_nodes
        assert "report" in plan.original_nodes

    def test_explain_shows_edges(self):
        """FlowPlan shows edges connecting sequential steps."""
        steps = [_make_step("a"), _make_step("b"), _make_step("c")]
        wf = Workflow(name="pipeline", steps=steps)

        plan = wf.explain()
        # Should have edges: a→b, b→c
        assert ("a", "b") in plan.original_edges
        assert ("b", "c") in plan.original_edges


# ---------------------------------------------------------------------------
# Memory Tests
# ---------------------------------------------------------------------------


class TestWorkflowMemory:
    """Workflow handles memory=True by auto-creating TieredMemoryStore."""

    def test_memory_true_creates_tiered_store(self):
        """memory=True creates a TieredMemoryStore internally."""
        from loomable.flow.memory import TieredMemoryStore

        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step], memory=True, session_id="sess1")

        assert wf._memory is not None
        assert isinstance(wf._memory, TieredMemoryStore)

    def test_memory_false_no_store(self):
        """memory=False (default) does not create a memory store."""
        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step])

        assert wf._memory is None

    def test_memory_custom_store(self):
        """A custom MemoryStore instance is accepted directly."""
        from loomable.flow.memory import TieredMemoryStore

        custom_store = TieredMemoryStore(session_id="custom")
        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step], memory=custom_store)

        assert wf._memory is custom_store


# ---------------------------------------------------------------------------
# Runnable Protocol Tests
# ---------------------------------------------------------------------------


class TestWorkflowRunnableProtocol:
    """Workflow satisfies the Runnable protocol."""

    def test_workflow_has_arun(self):
        """Workflow has an arun method."""
        step = _make_step("step_one")
        wf = Workflow(name="test", steps=[step])
        assert hasattr(wf, "arun")
        assert callable(wf.arun)

    @pytest.mark.asyncio
    async def test_workflow_arun_accepts_context(self):
        """Workflow.arun accepts optional context parameter."""
        step = _make_step("step_one", output_text="ok")
        wf = Workflow(name="test", steps=[step])

        ctx = RunContext()
        result = await wf.arun("input", context=ctx)
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Repr Tests
# ---------------------------------------------------------------------------


class TestWorkflowRepr:
    """Workflow has informative repr."""

    def test_repr(self):
        """Repr shows name and step count."""
        steps = [_make_step("a"), _make_step("b")]
        wf = Workflow(name="my_wf", steps=steps)
        assert "my_wf" in repr(wf)
        assert "2" in repr(wf)
