# Feature: workflow-ergonomics, Property 11: Loop with steps compiles to sequential Workflow body
"""Property 11: Loop with steps compiles to sequential Workflow body.

For any Loop constructed with a `steps` parameter containing N composable
elements, the Loop SHALL execute all N elements in sequence on each iteration,
producing the same result as a Loop whose `body` is a Workflow containing
those same steps.

**Validates: Requirements 5.1, 5.2, 5.5**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput
from loomable.flow.loop import AlwaysOkVerifier, Loop
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

# Strategy: simple input values
simple_inputs = st.one_of(
    st.text(min_size=1, max_size=30),
    st.integers(min_value=-1000, max_value=1000),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level execution tracker used to verify all steps ran
_execution_log: list[str] = []


def make_tracking_step(name: str, suffix: str) -> Step:
    """Create a Step that appends its name to _execution_log and transforms input."""

    def execute(input: Any) -> str:  # noqa: A002
        _execution_log.append(name)
        return f"{input}_{suffix}"

    return Step(name=name, agent=execute)


@st.composite
def distinct_step_specs(draw: st.DrawFn) -> list[tuple[str, str]]:
    """Generate 1-4 (step_name, suffix) tuples with distinct names.

    Each tuple represents a step that transforms input by appending a suffix.
    """
    n = draw(st.integers(min_value=1, max_value=4))
    names = draw(
        st.lists(
            step_names.map(lambda s: f"s_{s}"),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    suffixes = [draw(st.text(min_size=1, max_size=10, alphabet="abcdefghij")) for _ in range(n)]
    return list(zip(names, suffixes))


# ---------------------------------------------------------------------------
# Property tests: Loop with steps compiles to sequential Workflow body
# ---------------------------------------------------------------------------


class TestLoopStepsCompilesToSequentialBody:
    """Loop(steps=[...]) produces the same result as Loop(body=Workflow(steps=[...]))."""

    @settings(max_examples=100, deadline=None)
    @given(specs=distinct_step_specs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_loop_steps_produces_same_output_as_loop_with_workflow_body(
        self,
        specs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """A Loop with `steps` parameter produces the same output as a Loop
        whose `body` is a Workflow containing those same steps."""
        global _execution_log

        steps_a = [make_tracking_step(name, suffix) for name, suffix in specs]
        steps_b = [make_tracking_step(name, suffix) for name, suffix in specs]

        # Loop constructed with steps= parameter
        loop_with_steps = Loop(
            steps=steps_a,
            verifier=AlwaysOkVerifier(),  # Run once
            max_iterations=1,
        )

        # Loop constructed with body= Workflow containing the same steps
        workflow_body = Workflow(name="_loop_body_equiv", steps=steps_b)
        loop_with_body = Loop(
            body=workflow_body,
            verifier=AlwaysOkVerifier(),  # Run once
            max_iterations=1,
        )

        # Execute both
        _execution_log = []
        result_steps = await loop_with_steps.arun(input_val)
        log_steps = list(_execution_log)

        _execution_log = []
        result_body = await loop_with_body.arun(input_val)
        log_body = list(_execution_log)

        # Both should produce the same output text
        assert result_steps.output.text() == result_body.output.text(), (
            f"steps loop output: {result_steps.output.text()!r} != "
            f"body loop output: {result_body.output.text()!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(specs=distinct_step_specs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_loop_steps_executes_all_steps_in_sequence(
        self,
        specs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """All N steps in the Loop's `steps` list are executed on each iteration,
        in declaration order."""
        global _execution_log

        steps = [make_tracking_step(name, suffix) for name, suffix in specs]
        loop = Loop(
            steps=steps,
            verifier=AlwaysOkVerifier(),  # Run once
            max_iterations=1,
        )

        _execution_log = []
        await loop.arun(input_val)

        # All step names should appear in the log
        expected_names = [name for name, _ in specs]
        assert _execution_log == expected_names, (
            f"Expected execution order {expected_names}, got {_execution_log}"
        )

    @settings(max_examples=100, deadline=None)
    @given(specs=distinct_step_specs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_loop_steps_sequential_data_flow(
        self,
        specs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """Steps execute sequentially: the final output from Loop(steps=...)
        matches the final output from Loop(body=Workflow(steps=...)), confirming
        the sequential data flow is equivalent."""
        steps_a = [make_tracking_step(name, suffix) for name, suffix in specs]
        steps_b = [make_tracking_step(name, suffix) for name, suffix in specs]

        loop_steps = Loop(
            steps=steps_a,
            verifier=AlwaysOkVerifier(),  # Run once
            max_iterations=1,
        )

        loop_body = Loop(
            body=Workflow(name="_body_equiv", steps=steps_b),
            verifier=AlwaysOkVerifier(),  # Run once
            max_iterations=1,
        )

        result_steps = await loop_steps.arun(input_val)
        result_body = await loop_body.arun(input_val)

        # The last step's suffix should appear in the output of both
        last_suffix = specs[-1][1]
        assert last_suffix in result_steps.output.text(), (
            f"Last step suffix {last_suffix!r} not found in steps loop output: "
            f"{result_steps.output.text()!r}"
        )

        # Both paths produce the same output
        assert result_steps.output.text() == result_body.output.text(), (
            f"Sequential data flow mismatch: steps={result_steps.output.text()!r} "
            f"vs body={result_body.output.text()!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(specs=distinct_step_specs(), input_val=simple_inputs)
    @pytest.mark.asyncio
    async def test_loop_steps_multiple_iterations_execute_all_steps_each_time(
        self,
        specs: list[tuple[str, str]],
        input_val: Any,
    ) -> None:
        """When the Loop iterates multiple times, all N steps execute on each iteration."""
        global _execution_log

        max_iter = 2

        steps = [make_tracking_step(name, suffix) for name, suffix in specs]

        # A verifier that always fails (forces max_iterations)
        def never_ok(output: AgentOutput, context: RunContext) -> bool:
            return False

        loop = Loop(
            steps=steps,
            verifier=never_ok,
            max_iterations=max_iter,
        )

        _execution_log = []
        await loop.arun(input_val)

        # Each iteration should execute all steps
        expected_names = [name for name, _ in specs]
        expected_log = expected_names * max_iter
        assert _execution_log == expected_log, (
            f"Expected {max_iter} iterations of {expected_names}, got {_execution_log}"
        )
