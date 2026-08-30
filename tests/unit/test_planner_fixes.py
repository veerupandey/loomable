"""Planner opt-in, shared JSON parse, planning_model alias."""

from __future__ import annotations

import json

import pytest

from loomable import Agent
from loomable.agent import ModelSpec
from loomable.agent.plan_parse import parse_plan_steps
from loomable.content import ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.kernel.planner import ExecutionPlan, Planner, TaskContext


class ScriptedProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._i = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._i >= len(self._responses):
            return ModelResponse(content="[]")
        resp = self._responses[self._i]
        self._i += 1
        return resp


def test_parse_plan_steps_json_and_bullets() -> None:
    assert parse_plan_steps('["A", "B"]') == ["A", "B"]
    assert parse_plan_steps("- one\n- two") == ["one", "two"]
    assert parse_plan_steps('{"plan_steps": ["X"]}') == ["X"]


def test_build_without_planner_kwarg_leaves_planner_none() -> None:
    provider = ScriptedProvider([ModelResponse(content='["step"]')])
    built = Agent(
        model=ModelSpec(provider="p", provider_impl=provider),
        capabilities=ModelCapabilities(),
    ).build()
    assert built.planner is None


@pytest.mark.asyncio
async def test_kernel_planner_parses_json_steps() -> None:
    provider = ScriptedProvider(
        [ModelResponse(content='["Search docs", "Summarize"]')]
    )
    from loomable.kernel.model_interface import ModelInterface

    iface = ModelInterface(providers={"p": provider}, default_provider="p")
    planner = Planner(iface, planning_model_id=None)
    plan = await planner.plan(TaskContext(task="research"))
    assert plan.steps == ["Search docs", "Summarize"]


@pytest.mark.asyncio
async def test_planning_model_alias_without_tiers() -> None:
    provider = ScriptedProvider(
        [ModelResponse(content=json.dumps(["Plan step"]))],
    )
    planner_provider = ScriptedProvider(
        [ModelResponse(content=json.dumps(["Kernel step"]))]
    )
    calls: list[str] = []

    class TrackingPlanner(Planner):
        async def plan(self, task: TaskContext) -> ExecutionPlan:
            calls.append("plan")
            return ExecutionPlan(steps=["from-kernel"])

    agent = Agent(
        model=ModelSpec(provider="default", provider_impl=provider),
        capabilities=ModelCapabilities(),
        planning_model="planner-tier",
        planner=TrackingPlanner(
            __import__(
                "loomable.kernel.model_interface", fromlist=["ModelInterface"]
            ).ModelInterface(
                providers={"planner-tier": planner_provider},
                default_provider="planner-tier",
            ),
            planning_model_id="planner-tier",
        ),
    )
    built = agent.build()
    assert built.planner is not None
    plan = await built.planner.plan(TaskContext(task="t"))
    assert plan.steps == ["from-kernel"]
