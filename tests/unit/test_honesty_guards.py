"""Fail-loud guards for former silent no-ops (greenfield honesty)."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec, Team
from loomable.agent.errors import AgentConfigError
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import KnowledgeMemory, Memory, UserMemory


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok", usage={"input_tokens": 1, "output_tokens": 1})


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Echo())


def test_knowledge_without_embedder_raises() -> None:
    with pytest.raises(AgentConfigError, match="knowledge= requires embedder"):
        Agent(model=_model(), knowledge=["doc"], modalities="text").build()


def test_memory_tool_without_note_store_raises() -> None:
    with pytest.raises(AgentConfigError, match="memory_tool=True requires note_store"):
        Agent(model=_model(), memory_tool=True, modalities="text").build()


def test_user_memory_tool_without_store_raises_at_compose() -> None:
    with pytest.raises(AgentConfigError, match="note_store= or embedder="):
        Memory.compose(user=UserMemory(auto_extract=True, memory_tool=True)).to_agent_kwargs()


def test_knowledge_memory_documents_without_embedder_raises() -> None:
    with pytest.raises(AgentConfigError, match="KnowledgeMemory\\(documents="):
        Memory.compose(knowledge=KnowledgeMemory(documents=["a"])).to_agent_kwargs()


def test_team_hard_true_on_coordinate_raises() -> None:
    a = Agent(model=_model(), role="A", modalities="text")
    b = Agent(model=_model(), role="B", modalities="text")
    with pytest.raises(AgentConfigError, match="hard=True"):
        Team(members=[a, b], model=_model(), mode="coordinate", hard=True)


def test_team_hard_true_on_route_raises() -> None:
    a = Agent(model=_model(), role="A", modalities="text")
    with pytest.raises(AgentConfigError, match="hard=True"):
        Team(members=[a], model=_model(), mode="route", hard=True)


@pytest.mark.asyncio
async def test_agent_mode_case_astream_raises() -> None:
    agent = Agent(model=_model(), mode="case", modalities="text", max_rounds=1, max_plan_steps=1)
    with pytest.raises(AgentConfigError, match="does not support astream"):
        async for _ in agent.astream("hi"):
            pass


def test_unknown_agent_mode_raises() -> None:
    with pytest.raises(AgentConfigError, match="unsupported"):
        Agent(model=_model(), mode="cas", modalities="text")


def test_checkpointer_without_case_mode_raises() -> None:
    with pytest.raises(AgentConfigError, match="checkpointer="):
        Agent(model=_model(), checkpointer=object(), modalities="text")


def test_invalid_dispatch_raises() -> None:
    with pytest.raises(AgentConfigError, match="dispatch must be"):
        Agent(model=_model(), mode="case", dispatch="fork", modalities="text")


@pytest.mark.asyncio
async def test_mount_case_omits_ndjson_stream() -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    from loomable import Case
    from loomable.serve import mount_case

    case = Case(
        model=_model(),
        goal="triage",
        board=True,
        dispatch="reuse",
        max_rounds=1,
        max_steps=1,
        modalities="text",
    )
    app = FastAPI()
    mount_case(app, case, prefix="/cases")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": "hello"}]}
            ]
        }
        missing = await client.post("/cases/run/stream", json=body)
        assert missing.status_code == 404

        ok = await client.post("/cases/run/events", json=body)
        assert ok.status_code == 200
        assert "text/event-stream" in ok.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_mount_case_mode_agent_omits_ndjson_stream() -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    from loomable.serve import mount_agent

    agent = Agent(
        model=_model(),
        mode="case",
        modalities="text",
        max_rounds=1,
        max_plan_steps=1,
    )
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": "hello"}]}
            ]
        }
        missing = await client.post("/agent/run/stream", json=body)
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_astream_with_tools_preserves_tool_loop() -> None:
    from loomable.agent import tool
    from loomable.kernel.models import StreamEvent, ToolCall

    class _StreamNoTools:
        """Provider with stream() that would skip tools if used."""

        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id="1", tool_name="ping", args={})],
                    usage={"input_tokens": 1, "output_tokens": 1},
                )
            return ModelResponse(
                content="pong-done",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        async def stream(self, request: ModelRequest):
            yield StreamEvent(kind="text", text="streamed-no-tools")
            yield StreamEvent(kind="end")

    @tool
    def ping() -> str:
        return "pong"

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_StreamNoTools()),
        tools=[ping],
        modalities="text",
        max_tool_iterations=6,
    ).build()
    chunks = [c async for c in agent.astream("hi")]
    text = "".join(
        c.delta.data.decode("utf-8")
        for c in chunks
        if getattr(c.delta, "data", None) is not None
    )
    assert "pong-done" in text
    assert "streamed-no-tools" not in text


def test_create_deep_agent_case_kwargs_without_mode_raises(tmp_path) -> None:
    from loomable.agent import create_deep_agent

    with pytest.raises(AgentConfigError, match="mode='case'"):
        create_deep_agent(
            _model(),
            workspace=tmp_path,
            web_search=False,
            url_fetch=False,
            citations=False,
            images=False,
            enable_task_tool=False,
            think_tool=False,
            use_llm_summarizer=False,
            modalities="text",
            checkpointer=object(),
        )


def test_case_from_agent_copies_require_tools_and_hitl() -> None:
    from loomable.case import Case

    def allow(_call: object) -> bool:
        return True

    agent = Agent(
        model=_model(),
        mode="case",
        modalities="text",
        require_tools=["search_docs"],
        strict_require_tools=True,
        require_confirmation=["run_python"],
        approver=allow,
        max_rounds=1,
        max_plan_steps=1,
        board=False,
    )
    case = Case.from_agent(agent)
    rt = case._kwargs.get("agent_runtime") or {}
    assert rt.get("require_tools") == ["search_docs"]
    assert rt.get("strict_require_tools") is True
    assert rt.get("require_confirmation") == ["run_python"]
    assert rt.get("approver") is allow


def test_agent_approver_kwarg_reaches_built_agent() -> None:
    def allow(_call: object) -> bool:
        return True

    built = Agent(model=_model(), approver=allow, modalities="text").build()
    assert built.approver is allow
