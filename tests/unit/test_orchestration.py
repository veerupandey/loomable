"""Unit tests for loomable.agent.orchestration.Orchestrator (task 8.1, Req 11).

These tests use lightweight stub agents exposing an async ``arun`` (matching the
``BuiltAgent.arun`` shape) so orchestration logic is validated without constructing
real kernel loops or providers.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from loomable.agent.builder import OrchestrationMode
from loomable.agent.orchestration import Orchestrator
from loomable.agent.run import RunResult
from loomable.content import AgentInput, AgentOutput, Text
from loomable.kernel.errors import SubagentError


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class FakeAgent:
    """A minimal stand-in for BuiltAgent with an async ``arun``."""

    def __init__(
        self,
        name: str,
        *,
        reply: str = "ok",
        delay: float = 0.0,
        fail: bool = False,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.name = name
        self.session = _Session(f"session-{name}")
        self._reply = reply
        self._delay = delay
        self._fail = fail
        self._usage = usage or {}
        self.calls = 0

    async def arun(self, input: AgentInput, *, output_schema=None) -> RunResult:  # noqa: A002
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.name} boom")
        return RunResult(
            output=AgentOutput(parts=[Text(self._reply)]),
            session_id=self.session.session_id,
            usage=dict(self._usage),
        )


def _input() -> AgentInput:
    return AgentInput.from_text("hello")


# ---------------------------------------------------------------------------
# PARALLEL
# ---------------------------------------------------------------------------


def test_parallel_runs_all_and_aggregates_keyed_results() -> None:
    a = FakeAgent("a", reply="A", usage={"input_tokens": 1, "output_tokens": 2})
    b = FakeAgent("b", reply="B", usage={"input_tokens": 3, "output_tokens": 4})
    orch = Orchestrator([a, b], OrchestrationMode.PARALLEL)

    result = asyncio.run(orch.run(_input()))

    # Every sub-agent ran exactly once.
    assert a.calls == 1
    assert b.calls == 1
    # Results are keyed by sub-agent id.
    assert set(result.sub_results) == {"a", "b"}
    assert isinstance(result.sub_results["a"], RunResult)
    assert isinstance(result.sub_results["b"], RunResult)
    # Aggregated output concatenates child text in sub-agent order.
    assert result.output.text() == "AB"
    # Usage is summed across children.
    assert result.usage == {"input_tokens": 4, "output_tokens": 6}


def test_parallel_fault_isolation_one_fails_others_succeed() -> None:
    a = FakeAgent("a", reply="A")
    b = FakeAgent("b", fail=True)
    c = FakeAgent("c", reply="C")
    orch = Orchestrator([a, b, c], OrchestrationMode.PARALLEL)

    result = asyncio.run(orch.run(_input()))

    assert isinstance(result.sub_results["a"], RunResult)
    assert isinstance(result.sub_results["c"], RunResult)
    # The failing child yields a SubagentError naming it; siblings still return.
    error = result.sub_results["b"]
    assert isinstance(error, SubagentError)
    assert error.subagent_id == "b"
    # Aggregation includes only the successful children's text.
    assert result.output.text() == "AC"


def test_parallel_executes_concurrently() -> None:
    delay = 0.2
    agents = [FakeAgent(f"a{i}", delay=delay) for i in range(3)]
    orch = Orchestrator(agents, OrchestrationMode.PARALLEL)

    start = time.perf_counter()
    result = asyncio.run(orch.run(_input()))
    elapsed = time.perf_counter() - start

    assert set(result.sub_results) == {"a0", "a1", "a2"}
    # Concurrent execution: wall-clock is far less than the serial sum (3 * delay).
    assert elapsed < delay * len(agents)


# ---------------------------------------------------------------------------
# ROUTE
# ---------------------------------------------------------------------------


def test_route_runs_exactly_one_default_first() -> None:
    a = FakeAgent("a", reply="A")
    b = FakeAgent("b", reply="B")
    orch = Orchestrator([a, b], OrchestrationMode.ROUTE)

    result = asyncio.run(orch.run(_input()))

    assert a.calls == 1
    assert b.calls == 0
    assert result.output.text() == "A"
    assert set(result.sub_results) == {"a"}


def test_route_uses_injected_router() -> None:
    a = FakeAgent("a", reply="A")
    b = FakeAgent("b", reply="B")
    orch = Orchestrator(
        [a, b], OrchestrationMode.ROUTE, router=lambda agents, _input: 1
    )

    result = asyncio.run(orch.run(_input()))

    assert a.calls == 0
    assert b.calls == 1
    assert result.output.text() == "B"
    assert set(result.sub_results) == {"b"}


# ---------------------------------------------------------------------------
# COORDINATE
# ---------------------------------------------------------------------------


def test_coordinate_without_leader_concatenates() -> None:
    a = FakeAgent("a", reply="A")
    b = FakeAgent("b", reply="B")
    orch = Orchestrator([a, b], OrchestrationMode.COORDINATE)

    result = asyncio.run(orch.run(_input()))

    assert a.calls == 1
    assert b.calls == 1
    assert result.output.text() == "AB"
    assert set(result.sub_results) == {"a", "b"}


def test_coordinate_with_leader_synthesizes() -> None:
    a = FakeAgent("a", reply="A")
    b = FakeAgent("b", reply="B")
    leader = FakeAgent("leader", reply="SYNTHESIS")
    orch = Orchestrator([a, b], OrchestrationMode.COORDINATE, leader=leader)

    result = asyncio.run(orch.run(_input()))

    # Children delegated to, leader synthesizes a single output.
    assert a.calls == 1
    assert b.calls == 1
    assert leader.calls == 1
    assert result.output.text() == "SYNTHESIS"
    assert result.session_id == "session-leader"
    # Child results remain attached for traceability.
    assert set(result.sub_results) == {"a", "b"}


def test_empty_sub_agents_raises() -> None:
    orch = Orchestrator([], OrchestrationMode.PARALLEL)
    with pytest.raises(ValueError):
        asyncio.run(orch.run(_input()))
