"""Unit tests for loomable.flow.loop.Loop.

Validates:
- Loop stops when verifier reports success (Req 5.2)
- Loop stops on iteration cap with metadata recorded (Req 5.3)
- Loop runs body exactly once when no verifier supplied (Req 5.4)
- Loop feeds failure detail forward for self-correction (Req 5.5)
- Loop is usable standalone as a Runnable (Req 5.6)
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.loop import (
    AlwaysOkVerifier,
    CallableVerifier,
    Loop,
    VerdictResult,
    Verifier,
)
from loomable.flow.runnable import FunctionRunnable, Runnable


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


class SucceedOnNthAttempt:
    """A verifier that reports success only on the Nth check."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        self._count += 1
        if self._count >= self._n:
            return VerdictResult(ok=True)
        return VerdictResult(ok=False, detail=f"not ready yet (attempt {self._count})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_stops_on_verifier_success():
    """Req 5.2: Loop stops when verifier reports success and exposes last output."""
    body = CountingBody()
    verifier = SucceedOnNthAttempt(n=2)  # Succeeds on 2nd iteration

    loop = Loop(body, verifier=verifier, max_iterations=5)
    result = await loop.arun("hello")

    assert body.call_count == 2
    assert result.output.text() == "attempt-2"
    assert result.metadata["loop_iterations"] == 2
    assert result.metadata["loop_verified"] is True
    assert "loop_stop" not in result.metadata


@pytest.mark.asyncio
async def test_loop_stops_on_cap_with_metadata():
    """Req 5.3: Loop stops at cap and records loop_stop='max_iterations'."""
    body = CountingBody()
    # Verifier that never succeeds
    verifier = SucceedOnNthAttempt(n=999)

    loop = Loop(body, verifier=verifier, max_iterations=3)
    result = await loop.arun("hello")

    assert body.call_count == 3
    assert result.output.text() == "attempt-3"
    assert result.metadata["loop_stop"] == "max_iterations"
    assert result.metadata["loop_iterations"] == 3
    assert result.metadata["loop_verified"] is False


@pytest.mark.asyncio
async def test_loop_runs_once_with_no_verifier():
    """Req 5.4: No verifier = runs body exactly once (AlwaysOkVerifier)."""
    body = CountingBody()

    loop = Loop(body, max_iterations=5)
    result = await loop.arun("hello")

    assert body.call_count == 1
    assert result.output.text() == "attempt-1"
    assert result.metadata["loop_iterations"] == 1
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_feeds_failure_detail_forward():
    """Req 5.5: Failure detail from verifier is available to next iteration."""
    body = CountingBody()
    verifier = SucceedOnNthAttempt(n=3)

    loop = Loop(body, verifier=verifier, max_iterations=5)
    result = await loop.arun("initial input")

    assert body.call_count == 3
    # First call gets the original input
    assert body.inputs[0] == "initial input"
    # Second call should include failure detail
    assert "not ready yet (attempt 1)" in str(body.inputs[1])
    # Third call should include failure detail from second attempt
    assert "not ready yet (attempt 2)" in str(body.inputs[2])


@pytest.mark.asyncio
async def test_loop_with_callable_verifier():
    """A callable verifier is adapted via CallableVerifier."""
    body = CountingBody()
    # Callable that returns True when the output contains "attempt-2"
    verifier_fn = lambda output, ctx: "attempt-2" in output.text()

    loop = Loop(body, verifier=verifier_fn, max_iterations=5)
    result = await loop.arun("go")

    assert body.call_count == 2
    assert result.output.text() == "attempt-2"
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_satisfies_runnable_protocol():
    """Req 5.6: Loop is a Runnable (protocol-compatible)."""
    body = CountingBody()
    loop = Loop(body, max_iterations=1)

    # Structural subtyping check
    assert isinstance(loop, Runnable)

    # Can be called via the Runnable interface
    result = await loop.arun("test", context=RunContext())
    assert isinstance(result, RunResult)


@pytest.mark.asyncio
async def test_loop_with_function_runnable_body():
    """Loop works with a FunctionRunnable as body (composability)."""
    call_log = []

    def double(input):
        call_log.append(input)
        return f"doubled: {input}"

    body = FunctionRunnable(double)
    loop = Loop(body, max_iterations=1)
    result = await loop.arun("hello")

    assert result.output.text() == "doubled: hello"
    assert call_log == ["hello"]


@pytest.mark.asyncio
async def test_loop_max_iterations_default():
    """Default max_iterations is 3."""
    body = CountingBody()
    # Never-passing verifier
    verifier = SucceedOnNthAttempt(n=999)

    loop = Loop(body, verifier=verifier)  # default max_iterations=3
    result = await loop.arun("hello")

    assert body.call_count == 3
    assert result.metadata["loop_stop"] == "max_iterations"


@pytest.mark.asyncio
async def test_loop_passes_context_to_body():
    """Loop passes the RunContext through to the body."""
    received_contexts = []

    class ContextCapturingBody:
        async def arun(self, input, *, context=None):
            received_contexts.append(context)
            return _make_result("done")

    body = ContextCapturingBody()
    ctx = RunContext(max_steps=10)
    loop = Loop(body, max_iterations=1)
    await loop.arun("input", context=ctx)

    assert len(received_contexts) == 1
    assert received_contexts[0] is ctx
