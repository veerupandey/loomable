"""Tests for high-level plan= API and shared plan flow helpers."""

from __future__ import annotations

import asyncio
import json

import pytest

from loomable.agent import Agent, AlwaysPlan, ModelSpec, always_plan
from loomable.agent.errors import AgentConfigError
from loomable.agent.reasoning import (
    format_worker_notes,
    parse_plan_steps,
)
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.kernel.models import ModelRequest, ModelResponse


class _RoleAwareProvider:
    def __init__(self) -> None:
        self.roles: list[str] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text = "\n".join(str(getattr(m, "content", m)) for m in (request.messages or []))
        lower = text.lower()
        if "planner for parallel subagents" in lower and "json array" in lower:
            self.roles.append("planner")
            return ModelResponse(
                content=json.dumps(["Cover buyers", "Cover pricing", "Cover launch plan"])
            )
        if "research notes from parallel workers" in lower:
            self.roles.append("synthesizer")
            return ModelResponse(content="FINAL CEO ANSWER")
        if "your assigned step" in lower:
            self.roles.append("worker")
            return ModelResponse(content="Worker note")
        self.roles.append("other")
        return ModelResponse(content="other")


def test_parse_plan_steps_json_and_fallback() -> None:
    assert parse_plan_steps('["A", "B"]') == ["A", "B"]
    assert parse_plan_steps("- A\n- B\n") == ["A", "B"]


def test_format_worker_notes_pairs_steps() -> None:
    notes = format_worker_notes(["buyers result", "price result"], ["Buyers", "Pricing"])
    assert "Step 1 — Buyers" in notes
    assert "buyers result" in notes
    assert "Step 2 — Pricing" in notes


def test_always_plan_helper() -> None:
    clf = always_plan()
    assert isinstance(clf, AlwaysPlan)
    assert clf.classify(None, has_tools=False) == RunStrategy.PLAN  # type: ignore[arg-type]


def test_plan_always_api_fans_out() -> None:
    provider = _RoleAwareProvider()
    agent = Agent(
        model=ModelSpec(provider="test", provider_impl=provider),
        plan="always",
    )
    result = asyncio.run(agent.arun("Do a multi-part launch plan"))
    assert result.metadata["run_strategy"] == "plan"
    assert result.metadata["plan_trigger"] == "forced"
    assert result.metadata["plan_workers"] == 3
    assert result.metadata["plan_steps"] == ["Cover buyers", "Cover pricing", "Cover launch plan"]
    assert "FINAL CEO ANSWER" in result.output.text()
    # probe planner + 3 workers + synthesizer (cached planner avoids second plan call)
    assert provider.roles.count("planner") == 1
    assert provider.roles.count("worker") == 3
    assert provider.roles.count("synthesizer") == 1


def test_plan_true_uses_heuristic_router() -> None:
    agent = Agent(
        model=ModelSpec(provider="test", provider_impl=_RoleAwareProvider()),
        plan=True,
    )
    built = agent.build()
    assert built.complexity_router is not None
    # Short FAQ stays single under heuristic.
    from loomable.content import AgentInput

    assert (
        built.complexity_router.classify(
            AgentInput.from_text("What is X in one short paragraph?"),
            has_tools=False,
        )
        == RunStrategy.SINGLE
    )


def test_plan_and_complexity_router_conflict() -> None:
    with pytest.raises(AgentConfigError):
        Agent(
            model=ModelSpec(provider="test", provider_impl=_RoleAwareProvider()),
            plan="always",
            complexity_router=ComplexityRouter(),
        ).build()


def test_empty_plan_falls_back_to_single() -> None:
    class EmptyPlanProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            text = "\n".join(str(getattr(m, "content", m)) for m in (request.messages or []))
            if "planner for parallel subagents" in text.lower():
                return ModelResponse(content="[]")
            return ModelResponse(content="SINGLE FALLBACK")

    agent = Agent(
        model=ModelSpec(provider="test", provider_impl=EmptyPlanProvider()),
        plan="always",
    )
    result = asyncio.run(agent.arun("Anything"))
    assert result.metadata.get("plan_fallback") == "empty_plan"
    assert result.metadata.get("run_strategy") == "single"
    assert "SINGLE FALLBACK" in result.output.text()
