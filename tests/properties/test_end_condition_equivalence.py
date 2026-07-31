# Feature: workflow-ergonomics, Property 12: end_condition equivalence with verifier
"""Property 12: end_condition equivalence with verifier.

For any callable predicate `f` and for any sequence of inputs, a Loop
constructed with `end_condition=f` SHALL terminate at the same iteration as
a Loop constructed with `verifier=CallableVerifier(adapted_f)` given the
same body and inputs.

The end_condition receives a RunResult and returns True to stop.
The verifier (CallableVerifier) receives (AgentOutput, RunContext) and returns
True when verification passes (= stop). Both return True to stop, but they
receive different arguments. The adaptation is: end_condition(run_result)
becomes verifier(output, context) where output = run_result.output.

**Validates: Requirements 5.6, 5.7**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.loop import CallableVerifier, Loop
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: iteration at which the predicate should trigger (1-based)
# Keep max_iterations reasonable to avoid slow tests
stop_at_iterations = st.integers(min_value=1, max_value=8)

# Strategy: max_iterations for the loop (must be >= 1)
max_iterations_st = st.integers(min_value=1, max_value=10)

# Strategy: simple string inputs to pass to the loop
input_strings = st.text(min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create an AgentOutput with a single text part."""
    return AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )


class IterationTrackingBody:
    """A body that outputs 'attempt-N' on each call for deterministic tracking."""

    def __init__(self) -> None:
        self.call_count = 0

    async def arun(self, input: Any, *, context: RunContext | None = None) -> RunResult:
        self.call_count += 1
        output = _make_output(f"attempt-{self.call_count}")
        return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestEndConditionEquivalence:
    """end_condition=f terminates at the same iteration as verifier=adapted(f)."""

    @settings(max_examples=100, deadline=None)
    @given(
        stop_at=stop_at_iterations,
        max_iter=max_iterations_st,
        input_val=input_strings,
    )
    @pytest.mark.asyncio
    async def test_end_condition_and_verifier_terminate_same_iteration(
        self,
        stop_at: int,
        max_iter: int,
        input_val: str,
    ) -> None:
        """A predicate that stops at iteration N produces identical termination
        behavior whether provided as end_condition or as an adapted verifier."""

        # The predicate: stop when output contains "attempt-{stop_at}"
        def end_cond(run_result: RunResult) -> bool:
            return f"attempt-{stop_at}" in run_result.output.text()

        # The equivalent verifier adaptation (mirrors what Loop does internally):
        # end_condition receives RunResult, verifier receives (AgentOutput, RunContext).
        # The adaptation wraps the output in a RunResult before calling the predicate.
        def adapted_verifier(output: AgentOutput, context: RunContext) -> bool:
            result = RunResult(output=output, session_id="")
            return end_cond(result)

        # Create two independent bodies so call counts are isolated
        body_ec = IterationTrackingBody()
        body_v = IterationTrackingBody()

        # Loop with end_condition
        loop_ec = Loop(body_ec, end_condition=end_cond, max_iterations=max_iter)

        # Loop with equivalent CallableVerifier
        loop_v = Loop(body_v, verifier=CallableVerifier(adapted_verifier), max_iterations=max_iter)

        # Execute both
        result_ec = await loop_ec.arun(input_val)
        result_v = await loop_v.arun(input_val)

        # Both should terminate at the same iteration count
        assert body_ec.call_count == body_v.call_count, (
            f"end_condition ran {body_ec.call_count} iterations but "
            f"verifier ran {body_v.call_count} iterations"
        )

        # Both should produce the same output text
        assert result_ec.output.text() == result_v.output.text()

        # Both should have the same loop metadata
        assert result_ec.metadata.get("loop_iterations") == result_v.metadata.get("loop_iterations")
        assert result_ec.metadata.get("loop_verified") == result_v.metadata.get("loop_verified")
        assert result_ec.metadata.get("loop_stop") == result_v.metadata.get("loop_stop")

    @settings(max_examples=100, deadline=None)
    @given(
        max_iter=st.integers(min_value=1, max_value=8),
        input_val=input_strings,
    )
    @pytest.mark.asyncio
    async def test_never_stop_predicate_hits_cap_equivalently(
        self,
        max_iter: int,
        input_val: str,
    ) -> None:
        """A predicate that never returns True causes both loop variants to
        hit max_iterations identically."""

        def never_stop(run_result: RunResult) -> bool:
            return False

        def adapted_never_stop(output: AgentOutput, context: RunContext) -> bool:
            result = RunResult(output=output, session_id="")
            return never_stop(result)

        body_ec = IterationTrackingBody()
        body_v = IterationTrackingBody()

        loop_ec = Loop(body_ec, end_condition=never_stop, max_iterations=max_iter)
        loop_v = Loop(body_v, verifier=CallableVerifier(adapted_never_stop), max_iterations=max_iter)

        result_ec = await loop_ec.arun(input_val)
        result_v = await loop_v.arun(input_val)

        # Both hit the cap
        assert body_ec.call_count == max_iter
        assert body_v.call_count == max_iter
        assert body_ec.call_count == body_v.call_count

        # Both have same stop metadata
        assert result_ec.metadata["loop_stop"] == "max_iterations"
        assert result_v.metadata["loop_stop"] == "max_iterations"
        assert result_ec.metadata["loop_verified"] is False
        assert result_v.metadata["loop_verified"] is False

    @settings(max_examples=100, deadline=None)
    @given(
        max_iter=st.integers(min_value=1, max_value=8),
        input_val=input_strings,
    )
    @pytest.mark.asyncio
    async def test_always_stop_predicate_terminates_first_iteration(
        self,
        max_iter: int,
        input_val: str,
    ) -> None:
        """A predicate that always returns True causes both loop variants to
        terminate after exactly one iteration."""

        def always_stop(run_result: RunResult) -> bool:
            return True

        def adapted_always_stop(output: AgentOutput, context: RunContext) -> bool:
            result = RunResult(output=output, session_id="")
            return always_stop(result)

        body_ec = IterationTrackingBody()
        body_v = IterationTrackingBody()

        loop_ec = Loop(body_ec, end_condition=always_stop, max_iterations=max_iter)
        loop_v = Loop(body_v, verifier=CallableVerifier(adapted_always_stop), max_iterations=max_iter)

        result_ec = await loop_ec.arun(input_val)
        result_v = await loop_v.arun(input_val)

        # Both stop at iteration 1
        assert body_ec.call_count == 1
        assert body_v.call_count == 1

        # Both report verified success
        assert result_ec.metadata["loop_iterations"] == 1
        assert result_v.metadata["loop_iterations"] == 1
        assert result_ec.metadata["loop_verified"] is True
        assert result_v.metadata["loop_verified"] is True
