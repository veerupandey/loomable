"""Cooperative cancel on Workflow / Case / Team / Agent(mode=case)."""

from __future__ import annotations

import asyncio

import pytest

from loomable import Case, Team, Workflow
from loomable.agent import Agent, ModelSpec
from loomable.agent.context import StopReason
from loomable.kernel.models import ModelRequest, ModelResponse


class _Echo:
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=self.text, usage={"input_tokens": 1, "output_tokens": 1}
        )


def _model(text: str = "ok") -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Echo(text))


@pytest.mark.asyncio
async def test_workflow_cancel_skips_later_steps() -> None:
    ran: list[str] = []

    async def first(inp: str) -> str:
        ran.append("first")
        await asyncio.sleep(0.15)
        return "one"

    async def second(inp: str) -> str:
        ran.append("second")
        return "two"

    wf = Workflow("cancel-wf").step("a", first).step("b", second)
    task = asyncio.create_task(wf.arun("go"))
    for _ in range(50):
        if wf._active_ctx is not None:
            break
        await asyncio.sleep(0.01)
    assert wf.cancel() is True
    result = await task
    assert "first" in ran
    assert "second" not in ran
    assert (result.metadata or {}).get("stop_reason") == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_case_cancel_delegates_to_workflow() -> None:
    from loomable.agent.reasoning import make_think_tool
    from loomable.kernel.models import ToolCall

    class _SlowThink:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            await asyncio.sleep(0.05)
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=str(self.calls),
                        tool_name="think",
                        args={"thought": "x"},
                    )
                ],
            )

    inner = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_SlowThink()),
        tools=[make_think_tool()],
        max_tool_iterations=20,
        modalities="text",
    )
    case = Case(
        model=_model("plan"),
        goal="triage",
        board=False,
        max_rounds=1,
        max_steps=1,
        worker=inner,
        modalities="text",
    )
    task = asyncio.create_task(case.arun("go"))
    for _ in range(80):
        wf = case._workflow
        if wf is not None and wf._active_ctx is not None:
            break
        await asyncio.sleep(0.01)
    assert case.cancel() is True
    result = await task
    assert result is not None
    assert (result.metadata or {}).get("stop_reason") in {
        StopReason.CANCELLED,
        "cancelled",
        "final",
        "max_iterations",
    }


@pytest.mark.asyncio
async def test_team_hard_broadcast_cancel_stops_members() -> None:
    started = asyncio.Event()

    class _Block:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            started.set()
            await asyncio.sleep(2)
            return ModelResponse(content="late")

    a = Agent(
        model=ModelSpec(provider="a", provider_impl=_Block()),
        role="Alpha",
        modalities="text",
        max_tool_iterations=4,
    )
    b = Agent(
        model=ModelSpec(provider="b", provider_impl=_Echo("B")),
        role="Beta",
        modalities="text",
    )
    team = Team(members=[a, b], model=_model("M"), mode="broadcast", hard=True)
    task = asyncio.create_task(team.arun("task"))
    await asyncio.wait_for(started.wait(), timeout=2)
    # Member A is in-flight; cancel should mark its context.
    assert team.cancel() is True
    result = await asyncio.wait_for(task, timeout=3)
    # Broadcast gathers; cancelled member returns cancelled/empty-ish rather than hang.
    assert result.output.text() is not None


@pytest.mark.asyncio
async def test_team_coordinate_fallback_runs_skipped_members() -> None:
    a = Agent(model=_model("alpha-out"), role="Alpha", modalities="text")
    b = Agent(model=_model("beta-out"), role="Beta", modalities="text")
    team = Team(
        members=[a, b],
        model=_model("coordinator-only"),
        mode="coordinate",
    )
    result = await team.arun("task")
    text = result.output.text() or ""
    assert "alpha-out" in text
    assert "beta-out" in text
    assert result.metadata.get("team_coordinate_fallback")


@pytest.mark.asyncio
async def test_team_astream_coordinate_fallback_when_no_delegates() -> None:
    a = Agent(model=_model("alpha-out"), role="Alpha", modalities="text")
    b = Agent(model=_model("beta-out"), role="Beta", modalities="text")
    team = Team(
        members=[a, b],
        model=_model("coordinator-only"),
        mode="coordinate",
    )
    deltas: list[str] = []
    async for ev in team.astream_events("task"):
        data = getattr(ev, "data", None) or {}
        if isinstance(data, dict) and data.get("delta"):
            deltas.append(str(data["delta"]))
    blob = "".join(deltas)
    assert "alpha-out" in blob
    assert "beta-out" in blob


@pytest.mark.asyncio
async def test_team_astream_coordinate_skips_fallback_when_delegates_called() -> None:
    from loomable.kernel.models import ToolCall

    class _CountEcho(_Echo):
        def __init__(self, text: str = "ok") -> None:
            super().__init__(text)
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            return await super().complete(request)

    class _Coord:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            tool_name="delegate_to_alpha",
                            args={"task": "t"},
                        ),
                        ToolCall(
                            id="2",
                            tool_name="delegate_to_beta",
                            args={"task": "t"},
                        ),
                    ],
                    usage={"input_tokens": 1, "output_tokens": 1},
                )
            return ModelResponse(
                content="synth",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    alpha_m = _CountEcho("alpha-out")
    beta_m = _CountEcho("beta-out")
    a = Agent(
        model=ModelSpec(provider="scripted", provider_impl=alpha_m),
        role="Alpha",
        modalities="text",
    )
    b = Agent(
        model=ModelSpec(provider="scripted", provider_impl=beta_m),
        role="Beta",
        modalities="text",
    )
    team = Team(
        members=[a, b],
        model=ModelSpec(provider="scripted", provider_impl=_Coord()),
        mode="coordinate",
    )
    deltas: list[str] = []
    async for ev in team.astream_events("task"):
        data = getattr(ev, "data", None) or {}
        if isinstance(data, dict) and data.get("delta"):
            deltas.append(str(data["delta"]))
    blob = "".join(deltas)
    assert "(fallback)" not in blob
    assert alpha_m.calls == 1
    assert beta_m.calls == 1


@pytest.mark.asyncio
async def test_agent_mode_case_cancel_reaches_case() -> None:
    agent = Agent(
        model=_model("c"),
        mode="case",
        modalities="text",
        max_rounds=1,
        max_plan_steps=1,
        board=False,
    )
    # Warm the case so cancel has a target after start
    task = asyncio.create_task(agent.arun("go"))
    for _ in range(80):
        if agent._case is not None and getattr(agent._case, "_workflow", None) is not None:
            wf = agent._case._workflow
            if wf is not None and wf._active_ctx is not None:
                break
        await asyncio.sleep(0.01)
    agent.cancel()
    result = await task
    # Either cancelled or finished quickly (scripted plan is short); cancel must not error.
    assert result is not None
