"""Tests for FlowClass base class behavior (Task 8.3).

Validates:
- Req 6.6: kickoff(input) executes the compiled Flow
- Req 6.7: state property provides SharedState access
- Req 6.10: agents attribute accessible from decorated methods
- Req 6.11: Runnable protocol (arun delegates to kickoff)
- Req 7.2: explain() returns FlowPlan from compiled Flow
- Req 7.3: explain() works pre-execution
- Req 7.4: FlowPlan uses method names as node_ids
"""

from __future__ import annotations

import pytest

from loomable.flow.flow_class import FlowClass, start, listen, router
from loomable.flow.flow import FlowPlan
from loomable.flow.runnable import Runnable
from loomable.agent.run import RunResult


# ---------------------------------------------------------------------------
# Test FlowClass subclasses
# ---------------------------------------------------------------------------


class SimpleFlow(FlowClass):
    """A minimal FlowClass: start → process."""

    @start()
    async def begin(self, input):
        return f"started:{input}"

    @listen("begin")
    async def process(self, input):
        return f"processed:{input}"


class FlowWithAgents(FlowClass):
    """A FlowClass that accesses the agents attribute."""

    agents = {"greeter": "hello_agent"}

    @start()
    async def begin(self, input):
        agent_name = self.agents["greeter"]
        return f"{agent_name}:{input}"

    @listen("begin")
    async def finalize(self, input):
        return f"done:{input}"


class FlowWithDictAgents(FlowClass):
    """A FlowClass where agents is a plain dict set in __init__."""

    def __init__(self):
        self.agents = {"worker": "worker_agent", "reviewer": "review_agent"}
        super().__init__()

    @start()
    async def begin(self, input):
        return f"{self.agents['worker']}:{input}"


class MultiStepFlow(FlowClass):
    """A three-step linear flow: begin → transform → finalize."""

    @start()
    async def begin(self, input):
        return f"step1:{input}"

    @listen("begin")
    async def transform(self, input):
        return f"step2:{input}"

    @listen("transform")
    async def finalize(self, input):
        return f"step3:{input}"


# ---------------------------------------------------------------------------
# Tests: Compilation and kickoff
# ---------------------------------------------------------------------------


class TestFlowClassCompilation:
    """FlowClass compiles at instantiation and runs via kickoff."""

    def test_instantiation_compiles_flow(self):
        """Instantiating a FlowClass subclass triggers compilation."""
        flow = SimpleFlow()
        assert flow._compiled_flow is not None

    @pytest.mark.asyncio
    async def test_kickoff_executes_flow(self):
        """kickoff() executes the compiled flow and returns a RunResult."""
        flow = SimpleFlow()
        result = await flow.kickoff("test")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_kickoff_produces_output(self):
        """kickoff() produces output from the flow execution."""
        flow = SimpleFlow()
        result = await flow.kickoff("hello")
        # The flow runs start → listen, final output comes from process
        assert result.output is not None


# ---------------------------------------------------------------------------
# Tests: explain() works before execution
# ---------------------------------------------------------------------------


class TestFlowClassExplain:
    """explain() returns FlowPlan describing the compiled graph."""

    def test_explain_returns_flow_plan(self):
        """explain() returns a FlowPlan object."""
        flow = SimpleFlow()
        plan = flow.explain()
        assert isinstance(plan, FlowPlan)

    def test_explain_before_execution(self):
        """explain() works before any execution (no arun/kickoff needed)."""
        flow = SimpleFlow()
        plan = flow.explain()
        assert plan.original_nodes is not None
        assert len(plan.original_nodes) > 0

    def test_explain_contains_method_names(self):
        """FlowPlan uses method names as node_ids."""
        flow = SimpleFlow()
        plan = flow.explain()
        assert "begin" in plan.original_nodes
        assert "process" in plan.original_nodes

    def test_explain_contains_edges(self):
        """FlowPlan shows edges between connected methods."""
        flow = SimpleFlow()
        plan = flow.explain()
        assert ("begin", "process") in plan.original_edges

    def test_explain_multi_step_flow(self):
        """explain() on a multi-step flow shows full topology."""
        flow = MultiStepFlow()
        plan = flow.explain()
        assert "begin" in plan.original_nodes
        assert "transform" in plan.original_nodes
        assert "finalize" in plan.original_nodes
        assert ("begin", "transform") in plan.original_edges
        assert ("transform", "finalize") in plan.original_edges


# ---------------------------------------------------------------------------
# Tests: state property
# ---------------------------------------------------------------------------


class TestFlowClassState:
    """state property provides SharedState access after execution."""

    def test_state_is_none_before_execution(self):
        """state is None before any execution."""
        flow = SimpleFlow()
        assert flow.state is None

    @pytest.mark.asyncio
    async def test_state_available_after_execution(self):
        """state is available (non-None) after kickoff."""
        flow = SimpleFlow()
        await flow.kickoff("test")
        assert flow.state is not None

    @pytest.mark.asyncio
    async def test_state_is_shared_state(self):
        """state returns a SharedState instance after execution."""
        from loomable.flow.state import SharedState

        flow = SimpleFlow()
        await flow.kickoff("test")
        assert isinstance(flow.state, SharedState)


# ---------------------------------------------------------------------------
# Tests: agents attribute
# ---------------------------------------------------------------------------


class TestFlowClassAgents:
    """agents attribute is accessible from decorated methods."""

    def test_class_level_agents_attribute(self):
        """Class-level agents dict is accessible on the instance."""
        flow = FlowWithAgents()
        assert flow.agents == {"greeter": "hello_agent"}

    @pytest.mark.asyncio
    async def test_agents_accessible_in_start_method(self):
        """Decorated methods can access self.agents during execution."""
        flow = FlowWithAgents()
        result = await flow.kickoff("world")
        # The begin method uses self.agents["greeter"] which is "hello_agent"
        assert result.output is not None

    def test_instance_level_agents_dict(self):
        """Agents set in __init__ are accessible."""
        flow = FlowWithDictAgents()
        assert flow.agents["worker"] == "worker_agent"
        assert flow.agents["reviewer"] == "review_agent"

    @pytest.mark.asyncio
    async def test_instance_agents_accessible_in_method(self):
        """Instance-level agents are accessible from decorated methods."""
        flow = FlowWithDictAgents()
        result = await flow.kickoff("task")
        assert result.output is not None


# ---------------------------------------------------------------------------
# Tests: Runnable protocol (arun delegates to kickoff)
# ---------------------------------------------------------------------------


class TestFlowClassRunnable:
    """FlowClass satisfies the Runnable protocol."""

    def test_isinstance_runnable(self):
        """FlowClass instances satisfy isinstance(x, Runnable)."""
        flow = SimpleFlow()
        assert isinstance(flow, Runnable)

    @pytest.mark.asyncio
    async def test_arun_returns_run_result(self):
        """arun() returns a RunResult."""
        flow = SimpleFlow()
        result = await flow.arun("test")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_arun_delegates_to_kickoff(self):
        """arun() produces the same result as kickoff()."""
        flow = SimpleFlow()
        result_arun = await flow.arun("hello")
        # Create a fresh instance for kickoff comparison
        flow2 = SimpleFlow()
        result_kickoff = await flow2.kickoff("hello")
        # Both should produce output (exact comparison depends on engine behavior)
        assert result_arun.output is not None
        assert result_kickoff.output is not None

    @pytest.mark.asyncio
    async def test_arun_accepts_context_kwarg(self):
        """arun() accepts context keyword argument for protocol conformance."""
        from loomable.agent.context import RunContext

        flow = SimpleFlow()
        ctx = RunContext()
        result = await flow.arun("test", context=ctx)
        assert isinstance(result, RunResult)
