"""Regression tests for remaining framework-audit bugfixes."""

from __future__ import annotations

import asyncio

import pytest

from loomable.agent import Agent, ModelSpec, Team
from loomable.agent.context import RunContext
from loomable.agent.events import Event
from loomable.flow.engines.hierarchical import HierarchicalEngine
from loomable.flow.engines.parallel import ParallelEngine
from loomable.flow.flow import Flow
from loomable.flow.nodes import Node
from loomable.flow.state import SharedState
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.flow.runnable import FunctionRunnable
from loomable.persist.checkpoint import InMemoryCheckpointer
from loomable.stream import (
    NODE_FINISHED,
    NODE_STARTED,
    RUN_FINISHED,
    RUN_STARTED,
    TEXT_MESSAGE_CONTENT,
)


class _Echo:
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=self.text,
            usage={"input_tokens": 1, "output_tokens": 1},
        )


class _RecordingEvents:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class _FakeFlow:
    def __init__(self, nodes: dict[str, Node], edges: list | None = None) -> None:
        self._nodes = nodes
        self._edges = edges or []


@pytest.mark.asyncio
async def test_parallel_engine_merges_state_updates() -> None:
    async def planner(inp):
        return {"plan_steps": ["a", "b"], "note": "from-planner"}

    nodes = {"planner": Node(node_id="planner", runnable=FunctionRunnable(planner))}
    flow = _FakeFlow(nodes)
    state = SharedState()
    ctx = RunContext()
    await ParallelEngine().run(flow, "go", state, ctx)
    assert state.get("plan_steps") == ["a", "b"]
    assert state.get("note") == "from-planner"


@pytest.mark.asyncio
async def test_parallel_engine_node_events_inside_worker() -> None:
    async def slow(inp):
        await asyncio.sleep(0.02)
        return "slow"

    async def fast(inp):
        await asyncio.sleep(0.001)
        return "fast"

    nodes = {
        "slow": Node(node_id="slow", runnable=FunctionRunnable(slow)),
        "fast": Node(node_id="fast", runnable=FunctionRunnable(fast)),
    }
    flow = _FakeFlow(nodes)
    state = SharedState()
    events = _RecordingEvents()
    ctx = RunContext()
    ctx.events = events  # type: ignore[assignment]
    await ParallelEngine().run(flow, "go", state, ctx)

    starts = [e for e in events.events if e.kind == "node_start"]
    ends = [e for e in events.events if e.kind == "node_end"]
    assert {e.attributes.get("node_id") for e in starts} == {"slow", "fast"}
    assert {e.attributes.get("node_id") for e in ends} == {"slow", "fast"}
    # Per-node duration should not both equal the same superstep wall clock.
    durations = sorted(e.duration_ms or 0 for e in ends)
    assert durations[0] < durations[1]


@pytest.mark.asyncio
async def test_hierarchical_engine_merges_worker_state_updates() -> None:
    async def worker(inp):
        return {"plan_steps": ["w1"], "worker_flag": True}

    async def manager(inp):
        return {"manager_flag": True}

    nodes = {
        "mgr": Node(node_id="mgr", runnable=FunctionRunnable(manager), manager=True),
        "w": Node(node_id="w", runnable=FunctionRunnable(worker)),
    }
    flow = _FakeFlow(nodes)
    state = SharedState()
    ctx = RunContext()
    await HierarchicalEngine().run(flow, "go", state, ctx)
    assert state.get("plan_steps") == ["w1"]
    assert state.get("worker_flag") is True
    assert state.get("manager_flag") is True


@pytest.mark.asyncio
async def test_flow_astream_events_binds_checkpoint_session() -> None:
    cp = InMemoryCheckpointer()

    async def step(inp):
        return "done"

    flow = Flow({"a": step}, engine="sequential", checkpointer=cp)

    # Flow constructed without session_id; stream session must bind checkpoints.
    assert flow._session_id is None
    types: list[str] = []
    async for ev in flow.astream_events("hi", session_id="stream-sess-9"):
        types.append(ev.type)
    assert RUN_STARTED in types
    assert RUN_FINISHED in types
    assert flow._session_id is None  # restored after stream
    saved = await cp.get("stream-sess-9")
    assert saved is not None
    assert saved.complete is True


@pytest.mark.asyncio
async def test_team_astream_events_hard_broadcast() -> None:
    a = Agent(
        model=ModelSpec(provider="a", provider_impl=_Echo("A")),
        role="Alpha",
        modalities="text",
    )
    b = Agent(
        model=ModelSpec(provider="b", provider_impl=_Echo("B")),
        role="Beta",
        modalities="text",
    )
    team = Team(
        members=[a, b],
        model=ModelSpec(provider="m", provider_impl=_Echo("M")),
        mode="broadcast",
        hard=True,
    )
    types: list[str] = []
    node_ids: list[str] = []
    texts: list[str] = []
    async for ev in team.astream_events("task", session_id="team-1"):
        types.append(ev.type)
        if ev.type == NODE_STARTED:
            node_ids.append(str(ev.data.get("node_id")))
        if ev.type == TEXT_MESSAGE_CONTENT:
            texts.append(str(ev.data.get("delta") or ""))
    assert RUN_STARTED in types
    assert NODE_STARTED in types
    assert NODE_FINISHED in types
    assert RUN_FINISHED in types
    assert set(node_ids) == {"Alpha", "Beta"}
    blob = "\n".join(texts)
    assert "Alpha" in blob and "Beta" in blob
    assert "A" in blob and "B" in blob


@pytest.mark.asyncio
async def test_team_astream_events_hard_sequential() -> None:
    a = Agent(
        model=ModelSpec(provider="a", provider_impl=_Echo("first")),
        role="One",
        modalities="text",
    )
    b = Agent(
        model=ModelSpec(provider="b", provider_impl=_Echo("second")),
        role="Two",
        modalities="text",
    )
    team = Team(
        members=[a, b],
        model=ModelSpec(provider="m", provider_impl=_Echo("M")),
        mode="sequential",
        hard=True,
    )
    types: list[str] = []
    async for ev in team.astream_events("task"):
        types.append(ev.type)
    assert types.count(NODE_STARTED) == 2
    assert types.count(NODE_FINISHED) == 2
    assert RUN_FINISHED in types
