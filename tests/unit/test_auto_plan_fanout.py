"""End-to-end: plan API → PLAN → planner fan-out → synthesizer."""

from __future__ import annotations

import asyncio
import json

from loomable.agent import Agent, ModelSpec
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest, ModelResponse


# Strong enough for PLAN score threshold (>=4): many step cues + questions.
TOUGH_TASK = (
    "Compare and analyze Python, Rust, and Go for building web APIs. "
    "Break down the work step by step. For each language cover performance, "
    "developer experience, and ecosystem. Decompose into multiple steps. "
    "First research, then compare, then conclude. "
    "What are the tradeoffs? What should a team pick?"
)


class _RoleAwareProvider:
    """Returns planner JSON, per-step worker answers, then a synthesis."""

    def __init__(self) -> None:
        self.roles: list[str] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text = "\n".join(str(getattr(m, "content", m)) for m in (request.messages or []))
        lower = text.lower()
        if "planner for parallel subagents" in lower and "json array" in lower:
            self.roles.append("planner")
            return ModelResponse(
                content=json.dumps(
                    [
                        "Analyze Python for web APIs",
                        "Analyze Rust for web APIs",
                        "Analyze Go for web APIs",
                        "Compare tradeoffs across the three",
                    ]
                )
            )
        if "research notes from parallel workers" in lower:
            self.roles.append("synthesizer")
            return ModelResponse(content="FINAL: Use Go for simple APIs, Rust for hot paths.")
        if "your assigned step" in lower:
            self.roles.append("worker")
            return ModelResponse(content="Worker done for assigned step.")
        self.roles.append("other")
        return ModelResponse(content="unexpected")


def test_router_selects_plan_for_tough_task() -> None:
    strategy = ComplexityRouter().classify(
        AgentInput.from_text(TOUGH_TASK), has_tools=False
    )
    assert strategy == RunStrategy.PLAN


def test_auto_plan_fans_out_workers_and_synthesizes() -> None:
    provider = _RoleAwareProvider()
    agent = Agent(
        model=ModelSpec(provider="test", provider_impl=provider),
        plan="always",
    )
    result = asyncio.run(agent.arun(TOUGH_TASK))

    assert result.metadata.get("run_strategy") == "plan"
    assert result.metadata.get("plan_trigger") == "forced"
    assert result.metadata.get("plan_workers") == 4
    assert len(result.metadata.get("plan_steps") or []) == 4
    assert "FINAL:" in result.output.text()
    assert provider.roles.count("planner") == 1
    assert provider.roles.count("worker") == 4
    assert provider.roles.count("synthesizer") == 1
    assert provider.roles[0] == "planner"
    assert provider.roles[-1] == "synthesizer"
    assert all(r == "worker" for r in provider.roles[1:-1])
