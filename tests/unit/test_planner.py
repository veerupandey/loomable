"""Unit tests for loomable.kernel.planner.

Validates Requirements 15.1, 15.2, 15.3, 15.4:
- Planner produces an execution plan (15.1)
- Uses planning model when configured (15.2)
- Falls back to primary model when no planning model configured (15.3)
- Raises PlanningModelError when planning model unavailable (15.4)
"""

from __future__ import annotations

import pytest

from loomable.kernel.errors import PlanningModelError
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """A fake model provider that records calls and returns canned responses."""

    def __init__(self, response_content: str = "Step 1\nStep 2\nStep 3") -> None:
        self.calls: list[ModelRequest] = []
        self.response_content = response_content

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(
            content=self.response_content,
            usage={"input_tokens": 10, "output_tokens": 20},
            metadata={"provider": "fake"},
        )


def make_model_interface(
    providers: dict[str, FakeProvider],
    default: str = "primary",
) -> ModelInterface:
    """Create a ModelInterface with the given fake providers."""
    return ModelInterface(providers=providers, default_provider=default)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskContext:
    """Tests for the TaskContext dataclass."""

    def test_defaults(self) -> None:
        ctx = TaskContext(task="do something")
        assert ctx.task == "do something"
        assert ctx.context == {}

    def test_with_context(self) -> None:
        ctx = TaskContext(task="plan", context={"key": "value"})
        assert ctx.context == {"key": "value"}


class TestExecutionPlan:
    """Tests for the ExecutionPlan dataclass."""

    def test_defaults(self) -> None:
        plan = ExecutionPlan()
        assert plan.steps == []
        assert plan.metadata == {}

    def test_with_data(self) -> None:
        plan = ExecutionPlan(steps=["a", "b"], metadata={"model": "gpt-4"})
        assert plan.steps == ["a", "b"]
        assert plan.metadata == {"model": "gpt-4"}


class TestPlannerPlanProducesExecutionPlan:
    """Req 15.1: Planner produces an execution plan for the agent task."""

    @pytest.mark.asyncio
    async def test_plan_returns_execution_plan(self) -> None:
        provider = FakeProvider("Step 1: Gather info\nStep 2: Execute")
        mi = make_model_interface({"primary": provider})
        planner = Planner(model_interface=mi)

        task = TaskContext(task="Build a house")
        result = await planner.plan(task)

        assert isinstance(result, ExecutionPlan)
        assert len(result.steps) == 2
        assert result.steps[0] == "Step 1: Gather info"
        assert result.steps[1] == "Step 2: Execute"

    @pytest.mark.asyncio
    async def test_plan_includes_metadata(self) -> None:
        provider = FakeProvider("Do the thing")
        mi = make_model_interface({"primary": provider})
        planner = Planner(model_interface=mi)

        result = await planner.plan(TaskContext(task="test"))

        assert "usage" in result.metadata
        assert result.metadata["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_plan_handles_empty_response(self) -> None:
        provider = FakeProvider("")
        mi = make_model_interface({"primary": provider})
        planner = Planner(model_interface=mi)

        result = await planner.plan(TaskContext(task="empty"))

        assert isinstance(result, ExecutionPlan)
        assert result.steps == []


class TestPlannerUsesPlanningModel:
    """Req 15.2: Uses planning model when configured."""

    @pytest.mark.asyncio
    async def test_invokes_planning_model_when_configured(self) -> None:
        primary = FakeProvider("primary response")
        planning = FakeProvider("planning response")
        mi = make_model_interface(
            {"primary": primary, "planning-tier": planning}
        )
        planner = Planner(model_interface=mi, planning_model_id="planning-tier")

        result = await planner.plan(TaskContext(task="complex task"))

        # Planning model should be invoked, not primary
        assert len(planning.calls) == 1
        assert len(primary.calls) == 0
        assert result.steps == ["planning response"]


class TestPlannerFallsToPrimaryModel:
    """Req 15.3: Falls back to primary model when no planning model configured."""

    @pytest.mark.asyncio
    async def test_invokes_primary_model_when_no_planning_model(self) -> None:
        primary = FakeProvider("primary plan\nstep 2")
        mi = make_model_interface({"primary": primary})
        planner = Planner(model_interface=mi, planning_model_id=None)

        result = await planner.plan(TaskContext(task="simple task"))

        assert len(primary.calls) == 1
        assert result.steps == ["primary plan", "step 2"]

    @pytest.mark.asyncio
    async def test_planning_model_id_property_is_none(self) -> None:
        mi = make_model_interface({"primary": FakeProvider()})
        planner = Planner(model_interface=mi)

        assert planner.planning_model_id is None


class TestPlannerRaisesPlanningModelError:
    """Req 15.4: Raises PlanningModelError when planning model unavailable."""

    @pytest.mark.asyncio
    async def test_raises_planning_model_error_when_unavailable(self) -> None:
        primary = FakeProvider()
        mi = make_model_interface({"primary": primary})
        # Configure a planning model that doesn't exist in providers
        planner = Planner(model_interface=mi, planning_model_id="nonexistent-model")

        with pytest.raises(PlanningModelError) as exc_info:
            await planner.plan(TaskContext(task="will fail"))

        assert exc_info.value.model_id == "nonexistent-model"
        assert "nonexistent-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_does_not_raise_when_primary_used(self) -> None:
        primary = FakeProvider("works fine")
        mi = make_model_interface({"primary": primary})
        planner = Planner(model_interface=mi)

        # Should not raise - uses primary model which exists
        result = await planner.plan(TaskContext(task="no planning model"))
        assert isinstance(result, ExecutionPlan)

    @pytest.mark.asyncio
    async def test_planning_model_error_preserves_cause(self) -> None:
        primary = FakeProvider()
        mi = make_model_interface({"primary": primary})
        planner = Planner(model_interface=mi, planning_model_id="missing-model")

        with pytest.raises(PlanningModelError) as exc_info:
            await planner.plan(TaskContext(task="test"))

        # The original ModelProviderError should be the cause
        assert exc_info.value.__cause__ is not None
