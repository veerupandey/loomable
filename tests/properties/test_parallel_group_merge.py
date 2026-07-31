# Feature: workflow-ergonomics, Property 5: Parallel_Group executes all steps and merges outputs by name
"""Property 5: Parallel_Group executes all steps and merges outputs by name.

For any Parallel_Group containing N steps with distinct names, after execution
the SharedState SHALL contain an output entry keyed by each step's name, and
all N steps SHALL have executed regardless of order.

**Validates: Requirements 3.2, 3.4**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.parallel_group import Parallel_Group
from loomable.flow.state import SharedState
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid non-empty step names (distinct characters suitable for identifiers)
step_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=20,
)

# Strategy: number of parallel steps (1 to 5 as specified)
num_steps = st.integers(min_value=1, max_value=5)

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=30),
    st.integers(min_value=-1000, max_value=1000),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_output_step(name: str, output_marker: str) -> Step:
    """Create a Step that produces a deterministic, uniquely identifiable output.

    The step transforms input by appending the output_marker so we can verify
    which step produced which result.
    """

    def transform(x: Any) -> str:
        return f"{x}::{output_marker}"

    return Step(name=name, agent=transform)


@st.composite
def distinct_step_configs(draw: st.DrawFn) -> list[tuple[str, str]]:
    """Generate a list of 1-5 (name, marker) tuples with distinct names.

    Each tuple represents a step configuration: the name will be the step's
    identifier and the marker will be the unique output it produces.
    """
    n = draw(num_steps)
    # Generate N distinct names
    names = draw(
        st.lists(
            step_names,
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Each step gets a unique marker based on its index
    configs = [(name, f"marker_{i}") for i, name in enumerate(names)]
    return configs


# ---------------------------------------------------------------------------
# Property tests: All steps execute and outputs are keyed by name
# ---------------------------------------------------------------------------


class TestParallelGroupMerge:
    """Parallel_Group executes all steps and merges outputs by step name."""

    @settings(max_examples=100, deadline=None)
    @given(configs=distinct_step_configs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_all_step_names_appear_in_sub_results(
        self,
        configs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """After execution, sub_results contains a key for each step's name."""
        steps = [make_output_step(name, marker) for name, marker in configs]
        pg = Parallel_Group(*steps)

        result = await pg.arun(input_val)

        # sub_results should contain all step names as keys
        assert result.sub_results is not None
        for name, _ in configs:
            assert name in result.sub_results, (
                f"Step '{name}' not found in sub_results keys: "
                f"{list(result.sub_results.keys())}"
            )

    @settings(max_examples=100, deadline=None)
    @given(configs=distinct_step_configs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_each_sub_result_contains_expected_output(
        self,
        configs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """Each step's sub_result contains the output that step would produce."""
        steps = [make_output_step(name, marker) for name, marker in configs]
        pg = Parallel_Group(*steps)

        result = await pg.arun(input_val)

        assert result.sub_results is not None
        for name, marker in configs:
            expected_output = f"{input_val}::{marker}"
            sub_result = result.sub_results[name]
            assert isinstance(sub_result, RunResult)
            assert sub_result.output.text() == expected_output, (
                f"Step '{name}' expected output '{expected_output}', "
                f"got '{sub_result.output.text()}'"
            )

    @settings(max_examples=100, deadline=None)
    @given(configs=distinct_step_configs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_no_steps_skipped(
        self,
        configs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """All N steps execute — the number of sub_results equals N."""
        steps = [make_output_step(name, marker) for name, marker in configs]
        pg = Parallel_Group(*steps)

        result = await pg.arun(input_val)

        assert result.sub_results is not None
        assert len(result.sub_results) == len(configs), (
            f"Expected {len(configs)} sub_results, got {len(result.sub_results)}"
        )

    @settings(max_examples=100, deadline=None)
    @given(configs=distinct_step_configs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_result_is_run_result(
        self,
        configs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """Parallel_Group.arun always returns a RunResult instance."""
        steps = [make_output_step(name, marker) for name, marker in configs]
        pg = Parallel_Group(*steps)

        result = await pg.arun(input_val)

        assert isinstance(result, RunResult)
