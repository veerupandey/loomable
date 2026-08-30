"""Unit tests for agent gap fixes (planner, plan+tools, astream+tools, Team.astream)."""

from __future__ import annotations

import uuid

import pytest

from loomable import Agent, Team, tool
from loomable.agent import ModelSpec
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput, ModelCapabilities
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    StreamEvent,
    ToolCall,
)
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext


class StreamingScriptedProvider:
    """Provider with stream() for tool-loop + final text streaming tests."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.stream_calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._index >= len(self._responses):
            return ModelResponse(content="fallback", tool_calls=[])
        resp = self._responses[self._index]
        self._index += 1
        return resp

    async def stream(self, request: ModelRequest):
        self.stream_calls += 1
        if self._index >= len(self._responses):
            yield StreamEvent(kind="end", usage={"input_tokens": 1, "output_tokens": 1})
            return
        resp = self._responses[self._index]
        self._index += 1
        if resp.tool_calls:
            for tc in resp.tool_calls:
                yield StreamEvent(kind="tool_call", tool_call=tc)
        if resp.content:
            for word in str(resp.content).split():
                yield StreamEvent(kind="text", text=word + " ")
        yield StreamEvent(
            kind="end",
            usage=dict(resp.usage or {"input_tokens": 1, "output_tokens": 2}),
        )


def _agent(provider, **kwargs) -> Agent:
    return Agent(
        model=ModelSpec(provider="p", provider_impl=provider),
        capabilities=ModelCapabilities(),
        **kwargs,
    )


class TestPlanPathFixes:
    @pytest.mark.asyncio
    async def test_run_plan_uses_kernel_planner_when_set(self):
        calls: list[str] = []

        class RecordingPlanner(Planner):
            async def plan(self, task: TaskContext) -> ExecutionPlan:
                calls.append(task.task)
                return ExecutionPlan(steps=["Alpha step"])

        provider = StreamingScriptedProvider(
            [
                ModelResponse(content="alpha done"),
                ModelResponse(content="FINAL answer"),
            ]
        )
        mi = ModelInterface(providers={"p": provider}, default_provider="p")
        planner = RecordingPlanner(mi)

        class ForcePlan(ComplexityRouter):
            def classify(self, agent_input: AgentInput, *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        agent = Agent(
            model=ModelSpec(provider="p", provider_impl=provider),
            capabilities=ModelCapabilities(),
            planner=planner,
            complexity_router=ForcePlan(),
        )
        result = await agent.arun("Plan this task")
        assert calls and "Plan this task" in calls[0]
        assert result.output.text()

    @pytest.mark.asyncio
    async def test_run_plan_worker_uses_tool_loop(self):
        tool_calls: list[int] = []

        @tool
        def add_one(x: int) -> int:
            """Add one."""
            tool_calls.append(x)
            return x + 1

        provider = StreamingScriptedProvider(
            [
                ModelResponse(content="", tool_calls=[ToolCall(id="1", tool_name="add_one", args={"x": 1})]),
                ModelResponse(content="2"),
                ModelResponse(content="done step"),
                ModelResponse(content="FINAL"),
            ]
        )

        class ForcePlan(ComplexityRouter):
            def classify(self, agent_input: AgentInput, *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        mi = ModelInterface(providers={"p": provider}, default_provider="p")
        planner = Planner(mi)

        async def fake_plan(task: TaskContext) -> ExecutionPlan:
            return ExecutionPlan(steps=["Use add_one on 1"])

        planner.plan = fake_plan  # type: ignore[method-assign]

        agent = Agent(
            model=ModelSpec(provider="p", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[add_one],
            planner=planner,
            complexity_router=ForcePlan(),
            max_tool_iterations=5,
        )
        result = await agent.arun("compute")
        assert result.output.text()
        assert tool_calls == [1]
        assert any(
            (getattr(getattr(a, "result", None), "metadata", None) or {}).get("tool_name")
            == "add_one"
            for a in (result.tool_activity or [])
        )


class TestAstreamWithTools:
    @pytest.mark.asyncio
    async def test_astream_yields_deltas_during_tool_loop(self):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        provider = StreamingScriptedProvider(
            [
                ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", tool_name="ping", args={})],
                ),
                ModelResponse(content="hello world"),
            ]
        )
        agent = _agent(provider, tools=[ping], max_tool_iterations=4)
        chunks = [c async for c in agent.build().astream("go")]
        text = "".join(
            c.delta.data.decode("utf-8")
            for c in chunks
            if c.delta.data
        )
        assert "hello" in text or "world" in text
        assert provider.stream_calls >= 1
        assert chunks[-1].done is True


class TestTeamAstream:
    @pytest.mark.asyncio
    async def test_team_astream_delegates_to_coordinator(self):
        member = _agent(
            StreamingScriptedProvider([ModelResponse(content="member-ok")]),
            role="Worker",
        )
        coord = StreamingScriptedProvider([ModelResponse(content="coord-ok")])
        team = Team(
            members=[member],
            model=ModelSpec(provider="p", provider_impl=coord),
            mode="route",
        )
        chunks = [c async for c in team.astream("task")]
        assert chunks
        assert chunks[-1].done is True


class TestSandboxProfile:
    def test_create_deep_agent_sandbox_profile_enables_exec(self, tmp_path):
        from loomable.agent.deep import create_deep_agent

        agent = create_deep_agent(
            model=ModelSpec(
                provider="p",
                provider_impl=StreamingScriptedProvider([ModelResponse(content="x")]),
            ),
            profile="sandbox",
            workspace=str(tmp_path),
        )
        built = agent.build()
        tools = set(built.tool_runtime._tools.keys())
        assert "run_python" in tools or "run_shell" in tools
