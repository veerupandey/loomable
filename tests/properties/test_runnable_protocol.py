# Feature: workflow-ergonomics, Property 4: All new classes satisfy the Runnable protocol
"""Property 4: All new classes satisfy the Runnable protocol.

For any instance of Step, Workflow, Condition, Parallel_Group, or FlowClass,
calling `arun(input)` SHALL return a `RunResult` object, and the instance SHALL
satisfy `isinstance(instance, Runnable)`.

**Validates: Requirements 1.2, 2.6, 3.3, 4.4, 6.11, 9.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.flow.condition import Condition
from loomable.flow.flow_class import FlowClass, start, listen
from loomable.flow.parallel_group import Parallel_Group
from loomable.flow.runnable import Runnable
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid non-empty step names
step_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=20,
)

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(max_size=30),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.none(),
)

# Strategy: suffix strings for deterministic transforms
suffixes = st.text(min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(name: str, suffix: str) -> Step:
    """Create a Step wrapping a simple callable."""

    def transform(x: Any) -> str:
        return f"{x}_{suffix}"

    return Step(name=name, agent=transform)


def _make_workflow(step_names_list: list[str], suffix: str) -> Workflow:
    """Create a Workflow with N uniquely-named Steps."""
    steps = [_make_step(n, suffix) for n in step_names_list]
    return Workflow(name="test_workflow", steps=steps)


def _make_condition(suffix: str) -> Condition:
    """Create a Condition with a simple predicate and then_steps."""
    then_step = _make_step("then_step", suffix)
    return Condition(
        condition=lambda state: True,
        then_steps=[then_step],
    )


def _make_parallel_group(step_names_list: list[str], suffix: str) -> Parallel_Group:
    """Create a Parallel_Group with N uniquely-named Steps."""
    steps = [_make_step(n, suffix) for n in step_names_list]
    return Parallel_Group(*steps)


class _TestFlowClass(FlowClass):
    """A fixed FlowClass subclass for testing the Runnable protocol."""

    @start()
    async def begin(self, input: Any) -> str:  # noqa: A002
        return f"flow_result_{input}"


# Strategy: generate unique step name lists (1-3 unique names)
unique_step_name_lists = st.lists(
    step_names,
    min_size=1,
    max_size=3,
    unique=True,
)


# ---------------------------------------------------------------------------
# Property tests: Step satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestStepRunnableProtocol:
    """Step instances satisfy isinstance(instance, Runnable) and arun → RunResult."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_is_instance_of_runnable(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Step satisfies isinstance(instance, Runnable)."""
        step = _make_step(name, suffix)
        assert isinstance(step, Runnable)

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_arun_returns_run_result(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Step.arun returns a RunResult with an output attribute."""
        step = _make_step(name, suffix)
        result = await step.arun(input_val)
        assert isinstance(result, RunResult)
        assert hasattr(result, "output")


# ---------------------------------------------------------------------------
# Property tests: Workflow satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestWorkflowRunnableProtocol:
    """Workflow instances satisfy isinstance(instance, Runnable) and arun → RunResult."""

    @settings(max_examples=100, deadline=None)
    @given(names=unique_step_name_lists, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_workflow_is_instance_of_runnable(
        self, names: list[str], suffix: str, input_val: Any
    ) -> None:
        """Workflow satisfies isinstance(instance, Runnable)."""
        workflow = _make_workflow(names, suffix)
        assert isinstance(workflow, Runnable)

    @settings(max_examples=100, deadline=None)
    @given(names=unique_step_name_lists, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_workflow_arun_returns_run_result(
        self, names: list[str], suffix: str, input_val: Any
    ) -> None:
        """Workflow.arun returns a RunResult with an output attribute."""
        workflow = _make_workflow(names, suffix)
        result = await workflow.arun(input_val)
        assert isinstance(result, RunResult)
        assert hasattr(result, "output")


# ---------------------------------------------------------------------------
# Property tests: Condition satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestConditionRunnableProtocol:
    """Condition instances satisfy isinstance(instance, Runnable) and arun → RunResult."""

    @settings(max_examples=100, deadline=None)
    @given(suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_condition_is_instance_of_runnable(
        self, suffix: str, input_val: Any
    ) -> None:
        """Condition satisfies isinstance(instance, Runnable)."""
        condition = _make_condition(suffix)
        assert isinstance(condition, Runnable)

    @settings(max_examples=100, deadline=None)
    @given(suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_condition_arun_returns_run_result(
        self, suffix: str, input_val: Any
    ) -> None:
        """Condition.arun returns a RunResult with an output attribute."""
        condition = _make_condition(suffix)
        result = await condition.arun(input_val)
        assert isinstance(result, RunResult)
        assert hasattr(result, "output")


# ---------------------------------------------------------------------------
# Property tests: Parallel_Group satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestParallelGroupRunnableProtocol:
    """Parallel_Group instances satisfy isinstance(instance, Runnable) and arun → RunResult."""

    @settings(max_examples=100, deadline=None)
    @given(names=unique_step_name_lists, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_parallel_group_is_instance_of_runnable(
        self, names: list[str], suffix: str, input_val: Any
    ) -> None:
        """Parallel_Group satisfies isinstance(instance, Runnable)."""
        pg = _make_parallel_group(names, suffix)
        assert isinstance(pg, Runnable)

    @settings(max_examples=100, deadline=None)
    @given(names=unique_step_name_lists, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_parallel_group_arun_returns_run_result(
        self, names: list[str], suffix: str, input_val: Any
    ) -> None:
        """Parallel_Group.arun returns a RunResult with an output attribute."""
        pg = _make_parallel_group(names, suffix)
        result = await pg.arun(input_val)
        assert isinstance(result, RunResult)
        assert hasattr(result, "output")


# ---------------------------------------------------------------------------
# Property tests: FlowClass satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestFlowClassRunnableProtocol:
    """FlowClass instances satisfy isinstance(instance, Runnable) and arun → RunResult."""

    @settings(max_examples=100, deadline=None)
    @given(input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_flowclass_is_instance_of_runnable(
        self, input_val: Any
    ) -> None:
        """FlowClass satisfies isinstance(instance, Runnable)."""
        flow = _TestFlowClass()
        assert isinstance(flow, Runnable)

    @settings(max_examples=100, deadline=None)
    @given(input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_flowclass_arun_returns_run_result(
        self, input_val: Any
    ) -> None:
        """FlowClass.arun returns a RunResult with an output attribute."""
        flow = _TestFlowClass()
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)
        assert hasattr(result, "output")
