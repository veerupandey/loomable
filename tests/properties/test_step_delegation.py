# Feature: workflow-ergonomics, Property 1: Step delegation preserves output
"""Property 1: Step delegation preserves output.

For any valid Step wrapping either a Runnable or a callable, and for any input,
calling `Step.arun(input)` SHALL produce the same `RunResult.output` as calling
the wrapped agent's `arun(input)` directly (after FunctionRunnable adaptation
for callables).

**Validates: Requirements 1.2, 1.3**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid non-empty step names
step_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=30,
)

# Strategy: simple input values that functions can receive
simple_inputs = st.one_of(
    st.text(max_size=50),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(min_value=-100, max_value=100), max_size=5),
)

# Strategy: multiplier integers for deterministic transformation
multipliers = st.integers(min_value=-100, max_value=100)

# Strategy: suffix strings for string-based transformations
suffixes = st.text(min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleRunnable:
    """A minimal Runnable that transforms input deterministically."""

    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    async def arun(
        self, input: Any, *, context: Any = None  # noqa: A002
    ) -> RunResult:
        text = f"{input}_{self._suffix}"
        output = AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=text.encode("utf-8"),
                )
            ]
        )
        return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# Property tests: Step delegation with Runnable agents
# ---------------------------------------------------------------------------


class TestStepDelegationWithRunnable:
    """Step.arun produces the same output as the wrapped Runnable.arun directly."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_delegates_to_runnable_preserving_output(
        self,
        name: str,
        suffix: str,
        input_val: Any,
    ) -> None:
        """Wrapping a Runnable in a Step and calling arun produces the same
        output as calling the Runnable's arun directly."""
        runnable = SimpleRunnable(suffix)
        step = Step(name=name, agent=runnable)

        # Call Step.arun
        step_result = await step.arun(input_val)

        # Call Runnable.arun directly
        direct_result = await runnable.arun(input_val)

        # Both should produce the same output text
        assert step_result.output.text() == direct_result.output.text()

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_step_result_is_run_result(
        self,
        name: str,
        suffix: str,
        input_val: Any,
    ) -> None:
        """Step.arun always returns a RunResult instance."""
        runnable = SimpleRunnable(suffix)
        step = Step(name=name, agent=runnable)

        result = await step.arun(input_val)

        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Property tests: Step delegation with sync callables
# ---------------------------------------------------------------------------


class TestStepDelegationWithSyncCallable:
    """Step.arun with a sync callable produces the same output as
    FunctionRunnable(callable).arun directly."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, multiplier=multipliers, input_val=st.integers(min_value=-1000, max_value=1000))
    @pytest.mark.asyncio
    async def test_sync_callable_delegation_preserves_output(
        self,
        name: str,
        multiplier: int,
        input_val: int,
    ) -> None:
        """A sync callable wrapped in a Step produces the same output as
        FunctionRunnable wrapping the same callable."""

        def transform(x: Any) -> str:
            return f"result_{x}_{multiplier}"

        step = Step(name=name, agent=transform)

        # Call Step.arun
        step_result = await step.arun(input_val)

        # Call FunctionRunnable.arun directly (same adaptation Step does internally)
        fn_runnable = FunctionRunnable(transform)
        direct_result = await fn_runnable.arun(input_val)

        assert step_result.output.text() == direct_result.output.text()

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_sync_identity_callable_preserves_output(
        self,
        name: str,
        input_val: Any,
    ) -> None:
        """A sync identity callable wrapped in a Step produces the same output
        as FunctionRunnable wrapping it directly."""

        def identity(x: Any) -> Any:
            return x

        step = Step(name=name, agent=identity)
        fn_runnable = FunctionRunnable(identity)

        step_result = await step.arun(input_val)
        direct_result = await fn_runnable.arun(input_val)

        assert step_result.output.text() == direct_result.output.text()


# ---------------------------------------------------------------------------
# Property tests: Step delegation with async callables
# ---------------------------------------------------------------------------


class TestStepDelegationWithAsyncCallable:
    """Step.arun with an async callable produces the same output as
    FunctionRunnable(callable).arun directly."""

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, suffix=suffixes, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_async_callable_delegation_preserves_output(
        self,
        name: str,
        suffix: str,
        input_val: Any,
    ) -> None:
        """An async callable wrapped in a Step produces the same output as
        FunctionRunnable wrapping the same callable."""

        async def async_transform(x: Any) -> str:
            return f"async_{x}_{suffix}"

        step = Step(name=name, agent=async_transform)

        # Call Step.arun
        step_result = await step.arun(input_val)

        # Call FunctionRunnable.arun directly
        fn_runnable = FunctionRunnable(async_transform)
        direct_result = await fn_runnable.arun(input_val)

        assert step_result.output.text() == direct_result.output.text()

    @settings(max_examples=100, deadline=None)
    @given(name=step_names, input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_async_callable_returns_run_result(
        self,
        name: str,
        input_val: Any,
    ) -> None:
        """An async callable wrapped in a Step always returns a RunResult."""

        async def async_fn(x: Any) -> str:
            return f"output_{x}"

        step = Step(name=name, agent=async_fn)
        result = await step.arun(input_val)

        assert isinstance(result, RunResult)
