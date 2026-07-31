"""Unit tests for the consolidation task (15.2).

Verifies:
- Single BuiltAgent still works unchanged (non-regression).
- The removed imports (Pipeline, Orchestrator, OrchestrationMode, AutoPlan) are gone.
- Self-plan strategy uses the Flow engine when complexity router escalates to PLAN.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from loomable.agent import Agent, BuiltAgent, ModelSpec
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider that returns scripted responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses) if responses else ["ok"]
        self._call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return ModelResponse(content=self._responses[idx])


class _AlwaysPlanRouter:
    """A model classifier that always returns PLAN strategy."""

    def classify(self, agent_input, *, has_tools: bool) -> RunStrategy:
        return RunStrategy.PLAN


# ---------------------------------------------------------------------------
# 1. Single BuiltAgent still works (non-regression, Req 14.3)
# ---------------------------------------------------------------------------


class TestSingleAgentNonRegression:
    """BuiltAgent behaves unchanged for single-agent use after consolidation."""

    def test_single_shot_works(self):
        """A single agent with no tools runs single-shot and returns output."""
        provider = _FakeProvider(["Hello from the model!"])
        agent = Agent(model=ModelSpec(provider="test", provider_impl=provider))
        built = agent.build()

        result = asyncio.run(built.arun("Hi"))
        assert result.output.text() == "Hello from the model!"
        assert result.session_id is not None

    def test_tool_loop_works(self):
        """A single agent with tools runs the tool loop and returns output."""
        from loomable.agent.tools import FunctionTool

        provider = _FakeProvider(["No tools needed for this."])
        agent = Agent(model=ModelSpec(provider="test", provider_impl=provider))
        built = agent.build()

        # Add a tool so the tool loop is triggered
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = FunctionTool(greet, name="greet", description="Greet someone")
        built.tool_runtime._tools["greet"] = tool

        result = asyncio.run(built.arun("Say hello"))
        # The model returned no tool calls, so it exits immediately
        assert result.output.text() == "No tools needed for this."

    def test_built_agent_has_no_mode_or_sub_agents(self):
        """BuiltAgent no longer has mode, sub_agents, or max_plan_steps fields."""
        provider = _FakeProvider()
        built = Agent(model=ModelSpec(provider="test", provider_impl=provider)).build()

        assert not hasattr(built, "mode")
        assert not hasattr(built, "sub_agents")
        assert not hasattr(built, "max_plan_steps")


# ---------------------------------------------------------------------------
# 2. Removed imports are gone from __init__.py (Req 14.4, 14.6)
# ---------------------------------------------------------------------------


class TestRemovedImports:
    """Verify that Pipeline, Orchestrator, OrchestrationMode are no longer importable."""

    def test_pipeline_not_importable(self):
        import loomable.agent as agent_pkg

        assert not hasattr(agent_pkg, "Pipeline")

    def test_orchestrator_not_importable(self):
        import loomable.agent as agent_pkg

        assert not hasattr(agent_pkg, "Orchestrator")

    def test_orchestration_mode_not_importable(self):
        import loomable.agent as agent_pkg

        assert not hasattr(agent_pkg, "OrchestrationMode")

    def test_pipeline_module_gone(self):
        """The pipeline.py module itself should not exist."""
        with pytest.raises(ImportError):
            import importlib
            importlib.import_module("loomable.agent.pipeline")

    def test_orchestration_module_gone(self):
        """The orchestration.py module itself should not exist."""
        with pytest.raises(ImportError):
            import importlib
            importlib.import_module("loomable.agent.orchestration")

    def test_autoplan_module_gone(self):
        """The autoplan.py module itself should not exist."""
        with pytest.raises(ImportError):
            import importlib
            importlib.import_module("loomable.agent.autoplan")

    def test_remaining_exports_intact(self):
        """The other exports are still present and importable."""
        from loomable.agent import (
            Agent,
            BuiltAgent,
            ModelSpec,
            RunResult,
            RunContext,
            ComplexityRouter,
            RunStrategy,
            make_plan_tool,
            make_think_tool,
            FunctionTool,
        )

        # All these should import successfully
        assert Agent is not None
        assert BuiltAgent is not None
        assert ModelSpec is not None
        assert RunResult is not None
        assert RunContext is not None
        assert ComplexityRouter is not None
        assert RunStrategy is not None
        assert make_plan_tool is not None
        assert make_think_tool is not None
        assert FunctionTool is not None


# ---------------------------------------------------------------------------
# 3. Self-plan strategy uses the Flow engine (Req 17.2, 17.3)
# ---------------------------------------------------------------------------


class TestSelfPlanUsesFlow:
    """When the complexity router escalates to PLAN, the agent uses the Flow engine."""

    def test_plan_strategy_calls_run_plan(self):
        """When complexity router returns PLAN, _run_plan is invoked."""
        provider = _FakeProvider(["[\"Step 1\", \"Step 2\"]", "Result 1", "Result 2", "Final answer"])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            complexity_router=ComplexityRouter(model_classifier=_AlwaysPlanRouter()),
        )
        built = agent.build()

        # Patch _run_plan to verify it's called
        original_run_plan = built._run_plan
        plan_called = []

        async def _mock_run_plan(agent_input, *, output_schema=None, ctx=None, plan_trigger="router"):
            plan_called.append(True)
            return await original_run_plan(
                agent_input,
                output_schema=output_schema,
                ctx=ctx,
                plan_trigger=plan_trigger,
            )

        built._run_plan = _mock_run_plan

        result = asyncio.run(built.arun("Do something complex step by step and then compare"))
        assert len(plan_called) == 1

    def test_plan_uses_plan_and_execute_flow(self):
        """_run_plan internally uses plan_and_execute from loomable.flow.helpers."""
        provider = _FakeProvider([
            '[\"Research topic\", \"Write summary\"]',  # planner response
            "Research result",                           # worker step 1
            "Summary result",                           # worker step 2
            "Final synthesized answer",                 # synthesizer
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            complexity_router=ComplexityRouter(model_classifier=_AlwaysPlanRouter()),
        )
        built = agent.build()

        # Patch plan_and_execute at the helpers module level
        with patch("loomable.flow.helpers.plan_and_execute") as mock_pae:
            from loomable.agent.run import RunResult
            from loomable.content import AgentOutput, Text

            mock_flow = AsyncMock()
            mock_flow.arun.return_value = RunResult(
                output=AgentOutput(parts=[Text("Flow-based answer")]),
                session_id="test-session",
            )
            mock_pae.return_value = mock_flow

            result = asyncio.run(built.arun("Complex multi-step task"))

            # plan_and_execute was called
            mock_pae.assert_called_once()
            # The result came from the flow
            assert "Flow-based answer" in result.output.text()

    def test_make_plan_tool_uses_flow(self):
        """make_plan_tool builds a Flow via plan_and_execute (Req 17.2)."""
        from loomable.agent.reasoning import make_plan_tool

        provider = _FakeProvider([
            '[\"Step A\"]',    # planner
            "Step A done",     # worker
            "Final answer",    # synthesizer
        ])
        agent = Agent(model=ModelSpec(provider="test", provider_impl=provider))
        built = agent.build()

        plan_tool = make_plan_tool(built)
        assert plan_tool.name == "plan"

        # Patch plan_and_execute at the flow helpers module level
        with patch("loomable.flow.helpers.plan_and_execute") as mock_pae:
            from loomable.agent.run import RunResult
            from loomable.content import AgentOutput, Text

            mock_flow = AsyncMock()
            mock_flow.arun.return_value = RunResult(
                output=AgentOutput(parts=[Text("Plan tool flow result")]),
                session_id="test",
            )
            mock_pae.return_value = mock_flow

            # Invoke the plan tool
            result = asyncio.run(plan_tool._func("Do something complex"))

            mock_pae.assert_called_once()
            assert result == "Plan tool flow result"

    def test_default_routing_without_complexity_router(self):
        """Without a complexity router, agent uses tool-loop or single-shot (no PLAN)."""
        provider = _FakeProvider(["Direct answer"])
        agent = Agent(model=ModelSpec(provider="test", provider_impl=provider))
        built = agent.build()

        # No complexity router → no _run_plan path
        result = asyncio.run(built.arun("Simple question"))
        assert result.output.text() == "Direct answer"
