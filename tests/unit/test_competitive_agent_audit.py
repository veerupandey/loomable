"""Competitive audit probes: Agent / planner / todo / subagents / Team vs Agno-class surface."""

from __future__ import annotations

import pytest

from loomable import Agent, Team, tool
from loomable.agent import ModelSpec
from loomable.agent.errors import AgentConfigError
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput, ModelCapabilities
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext


class Scripted:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._i = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._i >= len(self._responses):
            return ModelResponse(content="fallback")
        r = self._responses[self._i]
        self._i += 1
        return r


def _spec(provider: Scripted) -> ModelSpec:
    return ModelSpec(provider="p", provider_impl=provider)


class TestPlannerParity:
    @pytest.mark.asyncio
    async def test_planning_model_id_wired_on_built_planner(self):
        mi = ModelInterface(
            providers={"p": Scripted([ModelResponse(content="x")])},
            default_provider="p",
        )
        planner = Planner(mi, planning_model_id="p")
        agent = Agent(
            model=_spec(Scripted([ModelResponse(content="hi")])),
            capabilities=ModelCapabilities(),
            planner=planner,
            planning_model="p",
        )
        built = agent.build()
        assert built.planner is planner
        assert built.planner.planning_model_id == "p"

    @pytest.mark.asyncio
    async def test_plan_path_fills_sub_results(self):
        class ForcePlan(ComplexityRouter):
            def classify(self, agent_input: AgentInput, *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        provider = Scripted(
            [
                ModelResponse(content="alpha"),
                ModelResponse(content="FINAL"),
            ]
        )
        mi = ModelInterface(providers={"p": provider}, default_provider="p")
        planner = Planner(mi)

        async def fake_plan(task: TaskContext) -> ExecutionPlan:
            return ExecutionPlan(steps=["Do alpha"])

        planner.plan = fake_plan  # type: ignore[method-assign]
        agent = Agent(
            model=_spec(provider),
            capabilities=ModelCapabilities(),
            planner=planner,
            complexity_router=ForcePlan(),
        )
        result = await agent.arun("task")
        assert result.sub_results
        assert "step_0" in result.sub_results


class TestPlanToolParity:
    @pytest.mark.asyncio
    async def test_plan_tool_worker_uses_tools(self):
        hits: list[int] = []

        @tool
        def bump(x: int) -> int:
            """Bump."""
            hits.append(x)
            return x + 1

        # Responses: planner JSON, worker tool call, worker final, synthesizer
        provider = Scripted(
            [
                ModelResponse(content='["Call bump"]'),
                ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id="1", tool_name="bump", args={"x": 3})],
                ),
                ModelResponse(content="bumped"),
                ModelResponse(content="SYNTH"),
            ]
        )
        agent = Agent(
            model=_spec(provider),
            capabilities=ModelCapabilities(),
            tools=[bump],
            plan_tool=True,
            max_tool_iterations=6,
        ).build()
        plan = agent.tool_runtime._tools["plan"]
        out = await plan._func("use bump")  # type: ignore[attr-defined]
        assert hits == [3]
        assert "SYNTH" in str(getattr(out, "content", out))


class TestTeamCompetitive:
    def test_tasks_mode_wires_todo_tools(self):
        member = Agent(
            model=_spec(Scripted([ModelResponse(content="m")])),
            role="Worker",
            capabilities=ModelCapabilities(),
        )
        team = Team(
            members=[member],
            model=_spec(Scripted([ModelResponse(content="c")])),
            mode="tasks",
            max_iterations=3,
        )
        built = team.agent.build()
        tools = set(built.tool_runtime._tools.keys())
        assert "write_todos" in tools
        assert "read_todos" in tools
        assert any(n.startswith("delegate_to_") for n in tools)
        assert team.agent._max_tool_iterations >= 12

    def test_hard_true_rejected_on_tasks(self):
        member = Agent(
            model=_spec(Scripted([ModelResponse(content="m")])),
            role="Worker",
            capabilities=ModelCapabilities(),
        )
        with pytest.raises(AgentConfigError, match="hard=True"):
            Team(
                members=[member],
                model=_spec(Scripted([ModelResponse(content="c")])),
                mode="tasks",
                hard=True,
            )

    def test_nested_team_as_member(self):
        leaf = Agent(
            model=_spec(Scripted([ModelResponse(content="leaf")])),
            role="Leaf",
            capabilities=ModelCapabilities(),
        )
        inner = Team(
            members=[leaf],
            model=_spec(Scripted([ModelResponse(content="inner")])),
            mode="route",
        )
        outer = Team(
            members=[inner],
            model=_spec(Scripted([ModelResponse(content="outer")])),
            mode="coordinate",
        )
        assert len(outer.members) == 1
        wrapped = outer.members[0]
        assert "NestedTeam" in (wrapped._role or wrapped._name or "")
        built = wrapped.build()
        assert "run_nested_team" in built.tool_runtime._tools

    @pytest.mark.asyncio
    async def test_hard_broadcast_sub_results_shape(self):
        a = Agent(
            model=_spec(Scripted([ModelResponse(content="A-ok")])),
            role="A",
            capabilities=ModelCapabilities(),
        )
        b = Agent(
            model=_spec(Scripted([ModelResponse(content="B-ok")])),
            role="B",
            capabilities=ModelCapabilities(),
        )
        team = Team(
            members=[a, b],
            model=_spec(Scripted([ModelResponse(content="unused")])),
            mode="broadcast",
        )
        result = await team.arun("go")
        text = result.output.text()
        assert "A-ok" in text and "B-ok" in text
        assert result.metadata.get("team_mode") == "broadcast"
