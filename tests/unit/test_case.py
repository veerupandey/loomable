"""Unit tests for Case, Board, and SharedState plan glue."""

from __future__ import annotations

import pytest

from loomable import Agent, Case, build_case_workflow
from loomable.agent import ModelSpec
from loomable.case import Board, parse_plan_steps
from loomable.flow.loop import VerdictResult
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.stream import NODE_STARTED, RUN_FINISHED, RUN_STARTED, STATE_DELTA, STATE_SNAPSHOT


class _Scripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        text_parts: list[str] = []
        for msg in request.messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text_parts.append(str(p.get("text", "")))
            elif isinstance(content, str):
                text_parts.append(content)
        blob = "\n".join(text_parts).lower()

        if "json array" in blob or "break the user task" in blob or "planner" in blob:
            return ModelResponse(
                content='["Gather ticket facts", "Assess SLA risk", "Draft SEV packet"]',
                usage={"input_tokens": 5, "output_tokens": 10},
            )
        if "verification failed" in blob or "failed verification" in blob or "failed acceptance" in blob:
            return ModelResponse(
                content="FINAL SEV-1 packet with evidence and next actions.",
                usage={"input_tokens": 5, "output_tokens": 8},
            )
        if "integrate" in blob or "synthesizer" in blob or "specialist/worker" in blob:
            if self.n < 6:
                return ModelResponse(
                    content="Draft packet without severity label yet.",
                    usage={"input_tokens": 5, "output_tokens": 6},
                )
            return ModelResponse(
                content="FINAL SEV-1 packet ready for war room.",
                usage={"input_tokens": 5, "output_tokens": 8},
            )
        return ModelResponse(
            content=f"done-step-{self.n}",
            usage={"input_tokens": 3, "output_tokens": 4},
        )


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Scripted())


def _sev_accept(output, context) -> VerdictResult:
    ok = "SEV-" in (output.text() or "")
    return VerdictResult(ok=ok, detail="missing SEV-" if not ok else "")


def test_parse_plan_steps_json_and_bullets() -> None:
    assert parse_plan_steps('["A", "B"]') == ["A", "B"]
    assert parse_plan_steps("- one\n- two") == ["one", "two"]


def test_board_lifecycle() -> None:
    board = Board()
    item = board.add("Triage INC")
    assert item.status == "open"
    board.update(item.id, status="in_progress")
    board.complete(item.id)
    assert board.list()[0].status == "done"


@pytest.mark.asyncio
async def test_function_runnable_dict_writes_state_updates() -> None:
    from loomable.flow.runnable import FunctionRunnable

    async def planner(inp):
        return {"plan_steps": ["a", "b"]}

    result = await FunctionRunnable(planner).arun("task")
    assert result.metadata["state_updates"]["plan_steps"] == ["a", "b"]
    assert result.structured["plan_steps"] == ["a", "b"]


@pytest.mark.asyncio
async def test_plan_and_execute_shared_state_glue() -> None:
    from loomable.flow.helpers import plan_and_execute

    async def planner(inp):
        return {"plan_steps": ["step-one", "step-two"]}

    async def worker(inp):
        return f"worked:{inp}"

    async def synthesizer(inp, *, context=None):
        pieces = []
        if context is not None and context.shared_state is not None:
            pieces = context.shared_state.get("map") or []
        return " | ".join(pieces)

    flow = plan_and_execute(planner, worker, synthesizer)
    result = await flow.arun("do it")
    assert "worked:step-one" in result.output.text()
    assert "worked:step-two" in result.output.text()


@pytest.mark.asyncio
async def test_plan_and_execute_empty_plan_publishes_empty_map() -> None:
    from loomable.flow.helpers import plan_and_execute

    async def planner(inp):
        return {"plan_steps": []}

    async def worker(inp):
        return f"worked:{inp}"

    async def synthesizer(inp, *, context=None):
        pieces = context.shared_state.get("map") if context else None
        assert pieces == []
        return "empty-ok"

    flow = plan_and_execute(planner, worker, synthesizer)
    result = await flow.arun("do it")
    assert "empty-ok" in result.output.text()


@pytest.mark.asyncio
async def test_case_dispatch_reuse_with_accept() -> None:
    case = Case(
        model=_model(),
        goal="Close INC-88421",
        board=True,
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=4,
        max_steps=3,
    )
    result = await case.arun("Handle INC-88421")
    assert result.metadata.get("case") is True
    assert "SEV-" in result.output.text()
    assert result.metadata.get("board")
    assert len(result.metadata["board"]["items"]) >= 1


@pytest.mark.asyncio
async def test_agent_mode_case() -> None:
    agent = Agent(
        model=_model(),
        mode="case",
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=4,
        max_plan_steps=3,
        modalities="text",
    )
    result = await agent.arun("Escalate INC-88421")
    assert result.metadata.get("case") is True
    assert "SEV-" in result.output.text()


@pytest.mark.asyncio
async def test_build_case_workflow() -> None:
    wf = build_case_workflow(model=_model(), dispatch="reuse", max_steps=2)
    assert wf.name == "case"
    result = await wf.arun("plan a caching strategy")
    assert result.output.text()


@pytest.mark.asyncio
async def test_case_astream_events_board_delta() -> None:
    case = Case(
        model=_model(),
        goal="Close INC",
        board=True,
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=4,
        max_steps=3,
    )
    types: list[str] = []
    async for ev in case.astream_events("Handle INC-88421"):
        types.append(ev.type)
    assert RUN_STARTED in types
    assert STATE_SNAPSHOT in types
    assert STATE_DELTA in types
    assert NODE_STARTED in types
    assert RUN_FINISHED in types


@pytest.mark.asyncio
async def test_workflow_astream_events_nodes() -> None:
    from loomable import Workflow

    async def step_a(inp):
        return "a-done"

    async def step_b(inp):
        return "b-done"

    wf = Workflow("pipe").step("a", step_a).step("b", step_b)
    types: list[str] = []
    async for ev in wf.astream_events("go"):
        types.append(ev.type)
    assert RUN_STARTED in types
    assert NODE_STARTED in types
    assert RUN_FINISHED in types


@pytest.mark.asyncio
async def test_agent_mode_case_reuses_board() -> None:
    """Board must survive across arun calls on the same Agent(mode=case)."""
    agent = Agent(
        model=_model(),
        mode="case",
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=4,
        max_plan_steps=3,
        modalities="text",
        board=True,
    )
    r1 = await agent.arun("Escalate INC-88421 first pass")
    board1 = (r1.metadata or {}).get("board") or {}
    assert board1.get("items")
    case = agent._get_case()
    n1 = len(case.board.list()) if case.board else 0
    r2 = await agent.arun("Escalate INC-88421 second pass")
    case2 = agent._get_case()
    assert case is case2
    assert case.board is case2.board
    assert n1 >= 1
    assert (r2.metadata or {}).get("case") is True


@pytest.mark.asyncio
async def test_workflow_state_without_caller_context() -> None:
    from loomable import Workflow

    async def step_a(inp):
        return "alpha"

    wf = Workflow("s").step("a", step_a)
    await wf.arun("x")
    assert wf.state.get("a") is not None


@pytest.mark.asyncio
async def test_case_board_rehydrates_from_checkpoint() -> None:
    from loomable.persist.checkpoint import Checkpoint, InMemoryCheckpointer

    board = Board()
    item = board.add("Resume triage INC-1")
    board.update(item.id, status="in_progress")
    cp = InMemoryCheckpointer()
    await cp.put(
        Checkpoint(
            thread_id="case-sess-1",
            step=1,
            session_state={"shared_state": {"board": board.to_dict()}},
            complete=False,
        )
    )
    case = Case(
        model=_model(),
        board=True,
        checkpointer=cp,
        session_id="case-sess-1",
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
    )
    assert case.board is not None
    assert case.board.list() == []
    await case._hydrate_board_from_checkpoint(resume=True)
    items = case.board.list()
    assert len(items) == 1
    assert items[0].title == "Resume triage INC-1"
    assert items[0].status == "in_progress"


@pytest.mark.asyncio
async def test_case_astream_events_hydrates_board_before_snapshot() -> None:
    from loomable.persist.checkpoint import Checkpoint, InMemoryCheckpointer
    from loomable.stream import STATE_SNAPSHOT

    board = Board()
    item = board.add("Keep this card")
    board.update(item.id, status="in_progress")
    cp = InMemoryCheckpointer()
    await cp.put(
        Checkpoint(
            thread_id="sse-sess",
            step=1,
            session_state={
                "shared_state": {
                    "board": board.to_dict(),
                    "plan_steps": ["Keep this card"],
                    "completed_node_ids": ["plan", "act"],
                },
                "completed_node_ids": ["plan", "act"],
            },
            complete=False,
        )
    )
    case = Case(
        model=_model(),
        board=True,
        checkpointer=cp,
        session_id="sse-sess",
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
    )
    snapshots = []
    async for ev in case.astream_events("continue", session_id="sse-sess", resume=True):
        if ev.type == STATE_SNAPSHOT:
            snapshots.append(ev.data.get("board") or {})
    assert snapshots
    assert snapshots[0].get("items")
    assert snapshots[0]["items"][0]["title"] == "Keep this card"


@pytest.mark.asyncio
async def test_case_board_rehydrates_from_complete_checkpoint() -> None:
    """Board should restore even when the latest checkpoint is complete=True."""
    from loomable.persist.checkpoint import Checkpoint, InMemoryCheckpointer

    board = Board()
    item = board.add("Closed card")
    board.update(item.id, status="done")
    cp = InMemoryCheckpointer()
    await cp.put(
        Checkpoint(
            thread_id="done-sess",
            step=9,
            session_state={"shared_state": {"board": board.to_dict()}},
            complete=True,
        )
    )
    case = Case(
        model=_model(),
        board=True,
        checkpointer=cp,
        session_id="done-sess",
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
    )
    await case._hydrate_board_from_checkpoint(resume=True)
    assert case.board is not None
    assert len(case.board.list()) == 1
    assert case.board.list()[0].status == "done"

    from loomable.persist.checkpoint import InMemoryCheckpointer

    cp = InMemoryCheckpointer()
    case = Case(
        model=_model(),
        board=True,
        checkpointer=cp,
        dispatch="reuse",
        accept=_sev_accept,
        max_rounds=2,
        max_steps=2,
        modalities="text",
    )
    case.bind_session("http-sess-9")
    wf = case.as_workflow()
    assert case._kwargs["session_id"] == "http-sess-9"
    assert wf._session_id == "http-sess-9"
    await case.arun("Escalate INC-88421 with SEV-1")
    saved = await cp.get("http-sess-9")
    assert saved is not None
