# Feature: workflow-ergonomics, Property 6: Parallel_Group isolates individual step failures
"""Property 6: Parallel_Group isolates individual step failures.

For any Parallel_Group containing N steps where one step raises an exception,
the remaining N-1 steps SHALL complete successfully and their outputs SHALL
appear in the merged SharedState.

**Validates: Requirements 3.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.flow.parallel_group import Parallel_Group
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: number of total steps (2 to 5)
total_steps_strategy = st.integers(min_value=2, max_value=5)

# Strategy: step name suffixes (distinct identifiers)
step_name_suffixes = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
)

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=30),
    st.integers(min_value=-1000, max_value=1000),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_good_step(name: str) -> Step:
    """Create a Step that succeeds and returns a predictable output."""

    def fn(input: Any) -> str:  # noqa: A002
        return f"output_from_{name}"

    return Step(name=name, agent=fn)


def make_failing_step(name: str) -> Step:
    """Create a Step that always raises an exception."""

    def fn(input: Any) -> str:  # noqa: A002
        raise RuntimeError(f"{name} exploded!")

    return Step(name=name, agent=fn)


# ---------------------------------------------------------------------------
# Composite strategy: generate N distinct step names then pick one as failing
# ---------------------------------------------------------------------------


@st.composite
def parallel_group_with_one_failure(draw: st.DrawFn):
    """Generate a Parallel_Group scenario with N steps (2-5), one of which fails.

    Returns (parallel_group, good_step_names, failing_step_name).
    """
    n = draw(total_steps_strategy)

    # Generate N distinct step names
    names = draw(
        st.lists(
            step_name_suffixes,
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Prefix to guarantee valid step names
    step_names = [f"step_{name}" for name in names]

    # Choose which index will be the failing step
    failing_index = draw(st.integers(min_value=0, max_value=n - 1))

    # Build the steps
    steps = []
    good_names = []
    failing_name = None

    for i, sname in enumerate(step_names):
        if i == failing_index:
            steps.append(make_failing_step(sname))
            failing_name = sname
        else:
            steps.append(make_good_step(sname))
            good_names.append(sname)

    pg = Parallel_Group(*steps)
    return pg, good_names, failing_name


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestParallelGroupFaultIsolation:
    """Property 6: Parallel_Group isolates individual step failures."""

    @settings(max_examples=100, deadline=None)
    @given(data=parallel_group_with_one_failure(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_non_failing_steps_complete_successfully(
        self,
        data: tuple[Parallel_Group, list[str], str],
        input_val: Any,
    ) -> None:
        """The N-1 non-failing steps all appear in sub_results with correct output."""
        pg, good_names, failing_name = data

        result = await pg.arun(input_val)

        # The result must have sub_results
        assert result.sub_results is not None

        # Each good step should appear in sub_results
        for name in good_names:
            assert name in result.sub_results, (
                f"Good step {name!r} missing from sub_results. "
                f"Available keys: {list(result.sub_results.keys())}"
            )

    @settings(max_examples=100, deadline=None)
    @given(data=parallel_group_with_one_failure(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_non_failing_steps_have_correct_output(
        self,
        data: tuple[Parallel_Group, list[str], str],
        input_val: Any,
    ) -> None:
        """Each non-failing step's output contains the expected value."""
        pg, good_names, failing_name = data

        result = await pg.arun(input_val)

        assert result.sub_results is not None

        for name in good_names:
            step_result = result.sub_results[name]
            # The good step produces "output_from_{name}"
            expected_text = f"output_from_{name}"
            assert expected_text in step_result.output.text(), (
                f"Expected {expected_text!r} in output for step {name!r}, "
                f"got {step_result.output.text()!r}"
            )

    @settings(max_examples=100, deadline=None)
    @given(data=parallel_group_with_one_failure(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_failing_step_does_not_prevent_others(
        self,
        data: tuple[Parallel_Group, list[str], str],
        input_val: Any,
    ) -> None:
        """The failing step does not prevent other steps from completing.

        We verify that ALL N-1 good steps produced results, meaning the
        failure was fully isolated.
        """
        pg, good_names, failing_name = data

        result = await pg.arun(input_val)

        assert result.sub_results is not None

        # Count successful steps (those without error metadata)
        successful_count = sum(
            1
            for name in good_names
            if name in result.sub_results
            and "error" not in result.sub_results[name].metadata
        )

        # All N-1 good steps should have succeeded
        assert successful_count == len(good_names), (
            f"Expected {len(good_names)} successful steps, got {successful_count}. "
            f"Failing step was {failing_name!r}."
        )

    @settings(max_examples=100, deadline=None)
    @given(data=parallel_group_with_one_failure(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_failing_step_recorded_in_sub_results(
        self,
        data: tuple[Parallel_Group, list[str], str],
        input_val: Any,
    ) -> None:
        """The failing step is still recorded in sub_results (with error metadata)."""
        pg, good_names, failing_name = data

        result = await pg.arun(input_val)

        assert result.sub_results is not None

        # The failing step should appear in sub_results with error metadata
        assert failing_name in result.sub_results, (
            f"Failing step {failing_name!r} not in sub_results."
        )
        failing_result = result.sub_results[failing_name]
        assert "error" in failing_result.metadata, (
            f"Failing step {failing_name!r} should have 'error' in metadata."
        )
