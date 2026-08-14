"""Parity: Agent / Team / Case / Flow share the same memory kwargs semantics."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec, NoteStore, Team
from loomable.agent.context import RunContext
from loomable.case import Case
from loomable.flow.workflow import Workflow
from loomable.kernel.long_term import LongTermStore, open_vector_store
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import open_session_store


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages)
        if "Alex" in blob and "What is my name" in blob:
            return ModelResponse(content="Your name is Alex.")
        return ModelResponse(content="ack")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 5), 1.0, 0.0]


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Echo())


@pytest.mark.asyncio
async def test_team_forwards_session_store_like_agent() -> None:
    store = open_session_store("memory")
    member = Agent(model=_model(), role="Worker", modalities="text")
    team = Team(
        [member],
        model=_model(),
        mode="coordinate",
        session_id="team-1",
        session_store=store,
        memory_window=4,
    )
    assert team.agent._session_store is store
    assert team.agent._session_id == "team-1"
    assert team.agent._memory_window == 4

    await team.agent.arun("My name is Alex")
    team.agent.bind_session("team-1")
    r2 = await team.agent.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_agent_bind_session_resumes_l1_l2() -> None:
    store = open_session_store("memory")
    a = Agent(
        model=_model(),
        session_id="http-1",
        session_store=store,
        modalities="text",
    )
    await a.arun("My name is Alex")

    a.bind_session("http-1")
    assert a._resume is True
    r2 = await a.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")

    a.bind_session("http-missing")
    assert a._resume is False


@pytest.mark.asyncio
async def test_case_from_agent_copies_note_store() -> None:
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    store = open_session_store("memory")
    agent = Agent(
        model=_model(),
        mode="case",
        session_id="case-mem",
        session_store=store,
        note_store=notes,
        memory_tool=True,
        modalities="text",
    )
    case = Case.from_agent(agent)
    mem = case._kwargs.get("agent_memory") or {}
    assert mem.get("note_store") is notes
    assert mem.get("session_store") is store
    assert mem.get("memory_tool") is True

    wf = case.as_workflow()
    assert wf is not None
    # Default planner was built with role-scoped session + shared note_store
    planner = case._kwargs.get("planner")
    # planner may be None until build — as_workflow already built defaults into steps
    assert case._kwargs["agent_memory"]["note_store"] is notes


@pytest.mark.asyncio
async def test_agent_in_workflow_forwards_run_context() -> None:
    """Agent.arun must accept Flow RunContext (same as BuiltAgent)."""
    store = open_session_store("memory")
    agent = Agent(
        model=_model(),
        session_id="flow-1",
        session_store=store,
        modalities="text",
    )

    seen: dict[str, object] = {}

    async def probe(inp, *, context=None):  # noqa: ANN001
        seen["ctx"] = context
        return await agent.arun(inp, context=context)

    wf = Workflow("mem-flow")
    wf.step("a", probe)
    result = await wf.arun("My name is Alex")
    assert result is not None
    assert isinstance(seen.get("ctx"), RunContext)

    agent2 = Agent(
        model=_model(),
        session_id="flow-1",
        session_store=store,
        resume=True,
        modalities="text",
    )
    r2 = await agent2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_case_role_scoped_session_ids() -> None:
    from loomable.agent.memory_opts import role_scoped_memory

    mem = role_scoped_memory(
        {"session_id": "c1", "session_store": "x", "note_store": "n"},
        role="planner",
    )
    assert mem["session_id"] == "c1:planner"
    assert mem["note_store"] == "n"
    assert "resume" not in mem
