# Feature: workflow-ergonomics, Property 3: Condition branches correctly based on predicate
"""Property 3: Condition branches correctly based on predicate.

For any Condition with a predicate, then_steps, and optional else_steps, and
for any SharedState: when the predicate returns True, only the then_steps SHALL
execute; when the predicate returns False and else_steps exist, only the
else_steps SHALL execute; when the predicate returns False and no else_steps
exist, the Condition SHALL pass input through unchanged.

**Validates: Requirements 2.3, 2.4, 2.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.condition import Condition
from loomable.flow.state import SharedState
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=50),
    st.integers(min_value=-1000, max_value=1000),
)

# Strategy: branch identifiers (used to tag which branch executed)
branch_tags = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=15,
)

# Strategy: number of steps in a branch (1 to 3 to keep tests fast)
branch_sizes = st.integers(min_value=1, max_value=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracking_step(tag: str, index: int) -> Step:
    """Create a Step that appends a unique marker to the output.

    The step transforms input by appending '[tag:index]' so we can verify
    which steps were executed and in what order.
    """

    def transform(x: Any) -> str:
        return f"{x}[{tag}:{index}]"

    return Step(name=f"step_{tag}_{index}", agent=transform)


def make_tracking_steps(tag: str, count: int) -> list[Step]:
    """Create a list of tracking steps that each append their tag and index."""
    return [make_tracking_step(tag, i) for i in range(count)]


def expected_output_for_branch(input_val: Any, tag: str, count: int) -> str:
    """Compute the expected output after running a branch of tracking steps.

    Each step appends '[tag:index]' to the running output.
    """
    result = str(input_val) if input_val is not None else ""
    for i in range(count):
        result = f"{result}[{tag}:{i}]"
    return result


# ---------------------------------------------------------------------------
# Property tests: Condition branches on True predicate
# ---------------------------------------------------------------------------


class TestConditionTrueBranch:
    """When predicate returns True, only then_steps execute."""

    @settings(max_examples=100, deadline=None)
    @given(
        input_val=simple_inputs,
        then_tag=branch_tags,
        then_count=branch_sizes,
    )
    @pytest.mark.asyncio
    async def test_true_predicate_executes_then_steps_only(
        self,
        input_val: Any,
        then_tag: str,
        then_count: int,
    ) -> None:
        """When condition returns True, only the then_steps branch runs."""
        then_steps = make_tracking_steps(then_tag, then_count)

        condition = Condition(
            condition=lambda state: True,
            then_steps=then_steps,
        )

        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await condition.arun(input_val, context=ctx)

        # The output should show the then_steps markers
        expected = expected_output_for_branch(input_val, then_tag, then_count)
        assert result.output.text() == expected

    @settings(max_examples=100, deadline=None)
    @given(
        input_val=simple_inputs,
        then_tag=branch_tags,
        else_tag=branch_tags,
        then_count=branch_sizes,
        else_count=branch_sizes,
    )
    @pytest.mark.asyncio
    async def test_true_predicate_does_not_execute_else_steps(
        self,
        input_val: Any,
        then_tag: str,
        else_tag: str,
        then_count: int,
        else_count: int,
    ) -> None:
        """When condition returns True with else_steps present,
        only then_steps execute (else_steps are not touched)."""
        then_steps = make_tracking_steps(then_tag, then_count)
        else_steps = make_tracking_steps(else_tag, else_count)

        condition = Condition(
            condition=lambda state: True,
            then_steps=then_steps,
            else_steps=else_steps,
        )

        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await condition.arun(input_val, context=ctx)

        # Output must contain then_tag markers, not else_tag markers
        output_text = result.output.text()
        expected = expected_output_for_branch(input_val, then_tag, then_count)
        assert output_text == expected

        # Verify else_tag markers are NOT present (unless tags happen to match)
        if then_tag != else_tag:
            assert f"[{else_tag}:" not in output_text


# ---------------------------------------------------------------------------
# Property tests: Condition branches on False predicate with else_steps
# ---------------------------------------------------------------------------


class TestConditionFalseBranchWithElse:
    """When predicate returns False and else_steps exist, only else_steps execute."""

    @settings(max_examples=100, deadline=None)
    @given(
        input_val=simple_inputs,
        then_tag=branch_tags,
        else_tag=branch_tags,
        then_count=branch_sizes,
        else_count=branch_sizes,
    )
    @pytest.mark.asyncio
    async def test_false_predicate_executes_else_steps_only(
        self,
        input_val: Any,
        then_tag: str,
        else_tag: str,
        then_count: int,
        else_count: int,
    ) -> None:
        """When condition returns False and else_steps exist, only
        the else_steps branch runs."""
        then_steps = make_tracking_steps(then_tag, then_count)
        else_steps = make_tracking_steps(else_tag, else_count)

        condition = Condition(
            condition=lambda state: False,
            then_steps=then_steps,
            else_steps=else_steps,
        )

        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await condition.arun(input_val, context=ctx)

        # Output must match else_steps execution
        output_text = result.output.text()
        expected = expected_output_for_branch(input_val, else_tag, else_count)
        assert output_text == expected

        # Verify then_tag markers are NOT present (unless tags happen to match)
        if then_tag != else_tag:
            assert f"[{then_tag}:" not in output_text


# ---------------------------------------------------------------------------
# Property tests: Condition with False predicate and no else_steps (passthrough)
# ---------------------------------------------------------------------------


class TestConditionFalseBranchPassthrough:
    """When predicate returns False and no else_steps, input passes through unchanged."""

    @settings(max_examples=100, deadline=None)
    @given(
        input_val=simple_inputs,
        then_tag=branch_tags,
        then_count=branch_sizes,
    )
    @pytest.mark.asyncio
    async def test_false_predicate_no_else_passes_input_through(
        self,
        input_val: Any,
        then_tag: str,
        then_count: int,
    ) -> None:
        """When condition returns False and no else_steps exist,
        the Condition passes input through unchanged."""
        then_steps = make_tracking_steps(then_tag, then_count)

        condition = Condition(
            condition=lambda state: False,
            then_steps=then_steps,
            else_steps=None,
        )

        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await condition.arun(input_val, context=ctx)

        # Output should be the string representation of the input (passthrough)
        expected = str(input_val) if input_val is not None else ""
        assert result.output.text() == expected

        # Verify then_tag markers are NOT present (then_steps did NOT execute)
        assert f"[{then_tag}:" not in result.output.text()

    @settings(max_examples=100, deadline=None)
    @given(
        input_val=simple_inputs,
        then_tag=branch_tags,
        then_count=branch_sizes,
    )
    @pytest.mark.asyncio
    async def test_false_predicate_no_else_returns_run_result(
        self,
        input_val: Any,
        then_tag: str,
        then_count: int,
    ) -> None:
        """Passthrough case still returns a proper RunResult."""
        then_steps = make_tracking_steps(then_tag, then_count)

        condition = Condition(
            condition=lambda state: False,
            then_steps=then_steps,
        )

        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await condition.arun(input_val, context=ctx)

        assert isinstance(result, RunResult)
        assert result.output is not None
