# Feature: workflow-ergonomics, Property 8: Workflow state reflects execution
"""Property 8: Workflow state reflects execution.

For any Workflow that has been executed, the `state` property SHALL reflect the
SharedState as written by the executed steps — i.e., for any step that writes a
key to SharedState, `workflow.state.get(key)` SHALL return the written value.

**Validates: Requirements 4.8**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.state import SharedState
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid non-empty step names (identifier-like characters)
step_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=20,
)

# Strategy: state keys (non-empty identifier-like strings)
state_keys = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=20,
)

# Strategy: state values (simple serializable types)
state_values = st.one_of(
    st.text(min_size=1, max_size=50),
    st.integers(min_value=-10000, max_value=10000),
    st.booleans(),
)

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=30),
    st.integers(min_value=-1000, max_value=1000),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state_writing_step(name: str, key: str, value: Any) -> Step:
    """Create a Step that writes a specific key/value to SharedState via context."""

    def write_to_state(input: Any, *, context: RunContext | None = None) -> str:  # noqa: A002
        if context is not None and context.shared_state is not None:
            context.shared_state.write(key, value)
        return f"wrote_{key}"

    return Step(name=name, agent=write_to_state)


@st.composite
def distinct_state_entries(draw: st.DrawFn) -> list[tuple[str, str, Any]]:
    """Generate 1-5 (step_name, state_key, state_value) tuples with distinct names and keys.

    Each tuple represents a step that will write a unique key to SharedState.
    Step names and state keys are drawn from disjoint pools to avoid collisions
    with the engine's own writes (which use node_id as the key).
    """
    n = draw(st.integers(min_value=1, max_value=5))
    # Generate N distinct step names with "step_" prefix
    names = draw(
        st.lists(
            step_names.map(lambda s: f"step_{s}"),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Generate N distinct state keys with "key_" prefix (disjoint from step names)
    keys = draw(
        st.lists(
            state_keys.map(lambda s: f"key_{s}"),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Generate N state values
    values = [draw(state_values) for _ in range(n)]
    return list(zip(names, keys, values))


# ---------------------------------------------------------------------------
# Property tests: Workflow state reflects execution
# ---------------------------------------------------------------------------


class TestWorkflowStateReflectsExecution:
    """After execution, workflow.state.get(key) returns the value written by each step."""

    @settings(max_examples=100, deadline=None)
    @given(entries=distinct_state_entries(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_state_contains_all_written_keys(
        self,
        entries: list[tuple[str, str, Any]],
        input_val: Any,
    ) -> None:
        """After execution, workflow.state contains every key written by steps."""
        steps = [
            make_state_writing_step(name, key, value)
            for name, key, value in entries
        ]
        workflow = Workflow(name="test_workflow", steps=steps)

        # Execute the workflow with a context so SharedState is captured
        context = RunContext()
        await workflow.arun(input_val, context=context)

        # Verify all written keys are accessible via workflow.state
        for _, key, _ in entries:
            assert workflow.state.get(key) is not None, (
                f"Key '{key}' not found in workflow.state after execution"
            )

    @settings(max_examples=100, deadline=None)
    @given(entries=distinct_state_entries(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_state_returns_correct_values(
        self,
        entries: list[tuple[str, str, Any]],
        input_val: Any,
    ) -> None:
        """After execution, workflow.state.get(key) returns the exact value written."""
        steps = [
            make_state_writing_step(name, key, value)
            for name, key, value in entries
        ]
        workflow = Workflow(name="test_workflow", steps=steps)

        context = RunContext()
        await workflow.arun(input_val, context=context)

        # Verify each key returns the expected value
        for _, key, expected_value in entries:
            actual = workflow.state.get(key)
            assert actual == expected_value, (
                f"Key '{key}': expected {expected_value!r}, got {actual!r}"
            )

    @settings(max_examples=100, deadline=None)
    @given(entries=distinct_state_entries(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_all_written_keys_accessible(
        self,
        entries: list[tuple[str, str, Any]],
        input_val: Any,
    ) -> None:
        """All keys written during execution are accessible — none are lost."""
        steps = [
            make_state_writing_step(name, key, value)
            for name, key, value in entries
        ]
        workflow = Workflow(name="test_workflow", steps=steps)

        context = RunContext()
        await workflow.arun(input_val, context=context)

        # Count how many keys are accessible
        accessible_count = sum(
            1 for _, key, _ in entries
            if workflow.state.get(key) is not None
        )
        assert accessible_count == len(entries), (
            f"Expected {len(entries)} accessible keys, got {accessible_count}"
        )
