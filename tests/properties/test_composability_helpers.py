# Feature: workflow-ergonomics, Property 15: New classes compose with existing helpers
"""Property 15: New classes compose with existing helpers.

For any Step, Workflow, or FlowClass instance, passing it to `sequential()`,
`parallel()`, or as a `Loop(body=...)` SHALL not raise an error and SHALL
produce a valid Flow that executes the instance as a node.

**Validates: Requirements 8.3, 9.1, 9.2**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.flow.flow import Flow
from loomable.flow.flow_class import FlowClass, start, listen
from loomable.flow.helpers import sequential, parallel
from loomable.flow.loop import Loop
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid non-empty step names (letters, numbers, dashes)
step_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=20,
)

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=30),
    st.integers(min_value=-1000, max_value=1000),
)

# Strategy: suffix strings for deterministic transformations
suffixes = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(name: str, suffix: str) -> Step:
    """Create a Step with a deterministic callable."""

    def transform(x: Any) -> str:
        return f"{x}_{suffix}"

    return Step(name=name, agent=transform)


def _make_workflow(name: str, suffix: str) -> Workflow:
    """Create a simple single-step Workflow."""
    step = _make_step(f"{name}_step", suffix)
    return Workflow(name=name, steps=[step])


def _make_flow_class(suffix: str) -> FlowClass:
    """Create a FlowClass instance with a single @start method."""

    class TestFlow(FlowClass):
        @start()
        async def begin(self, input: Any) -> str:  # noqa: A002
            return f"{input}_{suffix}"

    return TestFlow()


# ---------------------------------------------------------------------------
# Property tests: Step composes with existing helpers
# ---------------------------------------------------------------------------


class TestStepComposesWithHelpers:
    """Step instances can be passed to sequential(), parallel(), and Loop(body=...)."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_in_sequential(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Step to sequential() produces a valid Flow that executes."""
        step = _make_step(name, suffix)
        flow = sequential(step)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_in_parallel(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Step to parallel() produces a valid Flow that executes."""
        step = _make_step(name, suffix)
        flow = parallel(step)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_as_loop_body(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Step as Loop(body=...) produces a valid Loop that executes."""
        step = _make_step(name, suffix)
        loop = Loop(body=step, max_iterations=1)

        result = await loop.arun(input_val)
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Property tests: Workflow composes with existing helpers
# ---------------------------------------------------------------------------


class TestWorkflowComposesWithHelpers:
    """Workflow instances can be passed to sequential(), parallel(), and Loop(body=...)."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_workflow_in_sequential(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Workflow to sequential() produces a valid Flow that executes."""
        workflow = _make_workflow(name, suffix)
        flow = sequential(workflow)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_workflow_in_parallel(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Workflow to parallel() produces a valid Flow that executes."""
        workflow = _make_workflow(name, suffix)
        flow = parallel(workflow)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_workflow_as_loop_body(
        self, name: str, suffix: str, input_val: Any
    ) -> None:
        """Passing a Workflow as Loop(body=...) produces a valid Loop that executes."""
        workflow = _make_workflow(name, suffix)
        loop = Loop(body=workflow, max_iterations=1)

        result = await loop.arun(input_val)
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Property tests: FlowClass composes with existing helpers
# ---------------------------------------------------------------------------


class TestFlowClassComposesWithHelpers:
    """FlowClass instances can be passed to sequential(), parallel(), and Loop(body=...)."""

    @settings(max_examples=100, deadline=None)
    @given(suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_flowclass_in_sequential(
        self, suffix: str, input_val: Any
    ) -> None:
        """Passing a FlowClass to sequential() produces a valid Flow that executes."""
        fc = _make_flow_class(suffix)
        flow = sequential(fc)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_flowclass_in_parallel(
        self, suffix: str, input_val: Any
    ) -> None:
        """Passing a FlowClass to parallel() produces a valid Flow that executes."""
        fc = _make_flow_class(suffix)
        flow = parallel(fc)

        assert isinstance(flow, Flow)
        result = await flow.arun(input_val)
        assert isinstance(result, RunResult)

    @settings(max_examples=100, deadline=None)
    @given(suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_flowclass_as_loop_body(
        self, suffix: str, input_val: Any
    ) -> None:
        """Passing a FlowClass as Loop(body=...) produces a valid Loop that executes."""
        fc = _make_flow_class(suffix)
        loop = Loop(body=fc, max_iterations=1)

        result = await loop.arun(input_val)
        assert isinstance(result, RunResult)
