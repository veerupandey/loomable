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
