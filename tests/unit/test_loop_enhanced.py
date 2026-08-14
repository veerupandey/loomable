"""Unit tests for Loop enhancements: steps and verifier parameters.

Validates:
- Loop accepts ``steps=[...]`` compiled into a sequential Workflow body
- Loop accepts ``verifier=`` as Verifier or ``(output, context) -> bool``
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.flow.loop import Loop
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.step import Step


class CountingBody(FunctionRunnable):
    def __init__(self) -> None:
        self.call_count = 0
        super().__init__(self._run)

    async def _run(self, input):  # noqa: A002
        self.call_count += 1
        return f"attempt-{self.call_count}"


@pytest.mark.asyncio
async def test_loop_runs_once_without_verifier():
    body = CountingBody()
    loop = Loop(body, max_iterations=5)
    result = await loop.arun("go")
    assert body.call_count == 1
    assert result.output.text() == "attempt-1"


@pytest.mark.asyncio
async def test_loop_verifier_stops_loop():
    body = CountingBody()

    def should_stop(output: AgentOutput, context: RunContext) -> bool:
        return "attempt-2" in output.text()

    loop = Loop(body, verifier=should_stop, max_iterations=5)
    result = await loop.arun("go")

    assert body.call_count == 2
    assert result.output.text() == "attempt-2"
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_verifier_hits_cap():
    body = CountingBody()

    def never_stop(output: AgentOutput, context: RunContext) -> bool:
        return False

    loop = Loop(body, verifier=never_stop, max_iterations=3)
    result = await loop.arun("go")

    assert body.call_count == 3
    assert result.metadata["loop_stop"] == "max_iterations"
    assert result.metadata["loop_verified"] is False


@pytest.mark.asyncio
async def test_loop_verifier_with_steps():
    call_count = {"n": 0}

    def increment_fn(input):  # noqa: A002
        call_count["n"] += 1
        return f"count_{call_count['n']}"

    step = Step("counter", increment_fn)

    def stop_at_3(output: AgentOutput, context: RunContext) -> bool:
        return "count_3" in output.text()

    loop = Loop(steps=[step], verifier=stop_at_3, max_iterations=5)
    result = await loop.arun("start")

    assert call_count["n"] == 3
    assert result.metadata["loop_iterations"] == 3
    assert result.metadata["loop_verified"] is True


@pytest.mark.asyncio
async def test_loop_verifier_runs_body_once_on_immediate_true():
    body = CountingBody()

    def always_stop(output: AgentOutput, context: RunContext) -> bool:
        return True

    loop = Loop(body, verifier=always_stop, max_iterations=5)
    result = await loop.arun("go")

    assert body.call_count == 1
    assert result.metadata["loop_iterations"] == 1
    assert result.metadata["loop_verified"] is True
