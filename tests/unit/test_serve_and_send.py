"""Tests for Agno-ease serve + LangGraph-flex Send/map_over."""

from __future__ import annotations

import pytest

from loomable.flow.send import Send, send_args
from loomable.flow.state import SharedState


def test_send_args_extracts_payloads():
    items = [Send("worker", "a"), {"node": "w", "arg": "b"}, "plain"]
    assert send_args(items) == ["a", "b", "plain"]


@pytest.mark.asyncio
async def test_map_over_fan_out():
    from loomable import Workflow

    seen: list[str] = []

    async def worker(item: str) -> str:
        seen.append(item)
        return item.upper()

    wf = (
        Workflow("fan")
        .map_over(worker, over="tasks", name="map_tasks")
    )
    flow = wf._ensure_compiled()

    from loomable.agent.context import RunContext
    from loomable.content import AgentInput

    ctx = RunContext()
    ctx.shared_state = SharedState()
    ctx.shared_state.write(
        "tasks",
        [Send("worker", "one"), Send("worker", "two")],
    )
    result = await flow.arun(AgentInput.from_text("go"), context=ctx)
    assert sorted(seen) == ["one", "two"]
    assert result.output is not None


@pytest.mark.asyncio
async def test_workflow_bind_session_invalidates_compile():
    from loomable import Workflow

    async def step_fn(x: str) -> str:
        return x

    wf = Workflow("t", session_id="s1")
    wf.step("a", step_fn)
    flow1 = wf._ensure_compiled()
    wf.bind_session("s2")
    flow2 = wf._ensure_compiled()
    assert flow1 is not flow2
    assert wf._session_id == "s2"


@pytest.mark.asyncio
async def test_flow_astream_events_emits_run_paused():
    from loomable.agent.context import RunContext
    from loomable.content import AgentInput
    from loomable.flow.flow import Flow
    from loomable.flow.hitl import FlowPaused
    from loomable.flow.nodes import Node
    from loomable.flow.runnable import FunctionRunnable
    from loomable.persist.checkpoint import InMemoryCheckpointer, PendingAction
    from loomable.stream import RUN_PAUSED

    async def boom(_: str) -> str:
        raise FlowPaused(
            PendingAction(tool_name="risky", call_id="1", args={}),
            thread_id="t1",
        )

    flow = Flow(
        {"risky": Node(node_id="risky", runnable=FunctionRunnable(boom))},
        session_id="t1",
        checkpointer=InMemoryCheckpointer(),
    )
    events = [e async for e in flow.astream_events(AgentInput.from_text("x"))]
    assert any(getattr(e, "type", None) == RUN_PAUSED for e in events)


def test_mount_workflow_routes():
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    from loomable import Workflow
    from loomable.serve import mount_workflow

    async def echo(x: str) -> str:
        return f"echo:{x}"

    wf = Workflow("api", session_id="sess-1")
    wf.step("echo", echo)
    app = FastAPI()
    mount_workflow(app, wf, prefix="/wf")

    transport = httpx.ASGITransport(app=app)
    async def _check():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/wf/health")
            assert health.status_code == 200
            state = await client.get("/wf/state")
            assert state.status_code == 200
            run = await client.post(
                "/wf/run",
                json={"input_text": "hello", "session_id": "sess-1"},
            )
            assert run.status_code == 200
            assert "echo:hello" in run.json()["output"][0].get("text", "")

    import asyncio

    asyncio.run(_check())
