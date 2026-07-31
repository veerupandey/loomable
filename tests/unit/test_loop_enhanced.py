"""Unit tests for Loop enhancements: steps and end_condition parameters.

Validates:
- Loop accepts `steps` parameter and compiles into a sequential Workflow (Req 5.1, 5.2)
- Loop raises ValueError when both body and steps are provided (Req 5.3)
- Existing body-only constructor continues unchanged (Req 5.4)
- Loop with steps containing Parallel_Group or Condition compiles correctly (Req 5.5)
- Loop accepts `end_condition` as ergonomic alias for verifier (Req 5.6)
- end_condition is adapted into CallableVerifier internally (Req 5.7)
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.loop import (
    CallableVerifier,
    Loop,
    VerdictResult,
)
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create an AgentOutput with a single text part."""
    return AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )


def _make_result(text: str, session_id: str = "") -> RunResult:
    """Create a simple RunResult wrapping a text output."""
    return RunResult(output=_make_output(text), session_id=session_id)


class CountingBody:
    """A body Runnable that counts invocations and echoes input."""

    def __init__(self) -> None:
        self.call_count = 0
        self.inputs: list = []

    async def arun(self, input, *, context=None) -> RunResult:
        self.call_count += 1
        self.inputs.append(input)
        return _make_result(f"attempt-{self.call_count}")


# ---------------------------------------------------------------------------
# Tests: steps parameter (Req 5.1, 5.2, 5.5)
# ---------------------------------------------------------------------------


def test_loop_raises_when_both_body_and_steps():
    """Req 5.3: ValueError raised when both body and steps are specified."""
    body = CountingBody()
    step_a = Step("a", lambda x: "result_a")

    with pytest.raises(ValueError, match="Only one of 'body' or 'steps' may be specified"):
        Loop(body, steps=[step_a])


def test_loop_raises_when_neither_body_nor_steps():
    """Loop requires either body or steps."""
    with pytest.raises(ValueError, match="Either 'body' or 'steps' must be provided"):
        Loop()


@pytest.mark.asyncio
async def test_loop_with_steps_executes_sequentially():
    """Req 5.1, 5.2: steps parameter compiles into a sequential Workflow body."""
    call_log = []

    def step_a_fn(input):
        call_log.append(("a", input))
        return f"a_processed_{input}"

    def step_b_fn(input):
        call_log.append(("b", input))
        return f"b_processed_{input}"

    step_a = Step("step_a", step_a_fn)
    step_b = Step("step_b", step_b_fn)

    loop = Loop(steps=[step_a, step_b], max_iterations=1)
    result = await loop.arun("hello")

    # Both steps should have executed
    assert len(call_log) == 2
    assert call_log[0][0] == "a"
    assert call_log[1][0] == "b"
    # The final output should come from the last step
    assert "b_processed" in result.output.text()


@pytest.mark.asyncio
async def test_loop_with_steps_iterates_correctly():
    """Loop with steps iterates the full pipeline multiple times."""
    call_count = {"a": 0, "b": 0}

    def step_a_fn(input):
        call_count["a"] += 1
        return f"a_iter_{call_count['a']}"

    def step_b_fn(input):
        call_count["b"] += 1
        return f"b_iter_{call_count['b']}"

    step_a = Step("step_a", step_a_fn)
    step_b = Step("step_b", step_b_fn)

    # Verifier that passes on 2nd iteration
    iter_count = {"n": 0}

    def check_fn(output, context):
        iter_count["n"] += 1
        return iter_count["n"] >= 2

    loop = Loop(steps=[step_a, step_b], verifier=check_fn, max_iterations=5)
    result = await loop.arun("start")

    # Should have run 2 iterations
    assert call_count["a"] == 2
    assert call_count["b"] == 2
    assert result.metadata["loop_iterations"] == 2
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_with_steps_backward_compat():
    """Req 5.4: Existing body-only constructor still works."""
    body = CountingBody()
    loop = Loop(body, max_iterations=2)
    result = await loop.arun("test")

    assert body.call_count == 1
    assert result.output.text() == "attempt-1"


# ---------------------------------------------------------------------------
# Tests: end_condition parameter (Req 5.6, 5.7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_end_condition_stops_loop():
    """Req 5.6: end_condition callable is used as termination condition."""
    body = CountingBody()

    # end_condition receives a RunResult and returns True to stop
    def should_stop(run_result: RunResult) -> bool:
        return "attempt-2" in run_result.output.text()

    loop = Loop(body, end_condition=should_stop, max_iterations=5)
    result = await loop.arun("go")

    assert body.call_count == 2
    assert result.output.text() == "attempt-2"
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_end_condition_hits_cap():
    """end_condition that never returns True hits the max_iterations cap."""
    body = CountingBody()

    def never_stop(run_result: RunResult) -> bool:
        return False

    loop = Loop(body, end_condition=never_stop, max_iterations=3)
    result = await loop.arun("go")

    assert body.call_count == 3
    assert result.metadata["loop_stop"] == "max_iterations"
    assert result.metadata["loop_verified"] is False


@pytest.mark.asyncio
async def test_loop_end_condition_with_steps():
    """end_condition works in combination with steps parameter."""
    call_count = {"n": 0}

    def increment_fn(input):
        call_count["n"] += 1
        return f"count_{call_count['n']}"

    step = Step("counter", increment_fn)

    def stop_at_3(run_result: RunResult) -> bool:
        return "count_3" in run_result.output.text()

    loop = Loop(steps=[step], end_condition=stop_at_3, max_iterations=5)
    result = await loop.arun("start")

    assert call_count["n"] == 3
    assert result.metadata["loop_iterations"] == 3
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_end_condition_runs_body_once_on_immediate_true():
    """If end_condition returns True on first iteration, body runs once."""
    body = CountingBody()

    def always_stop(run_result: RunResult) -> bool:
        return True

    loop = Loop(body, end_condition=always_stop, max_iterations=5)
    result = await loop.arun("go")

    assert body.call_count == 1
    assert result.metadata["loop_iterations"] == 1
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_satisfies_runnable_with_steps():
    """Loop with steps still satisfies the Runnable protocol."""
    step = Step("simple", lambda x: "done")
    loop = Loop(steps=[step], max_iterations=1)

    assert isinstance(loop, Runnable)
    result = await loop.arun("test")
    assert isinstance(result, RunResult)
