"""End-to-end: ComplexityRouter → PLAN → planner fan-out → synthesizer."""

from __future__ import annotations

import asyncio
import json

from loomable.agent import Agent, ModelSpec
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest, ModelResponse


TOUGH_TASK = (
    "Compare and analyze Python, Rust, and Go for building web APIs, then "
    "break down the work step by step covering performance, developer "
    "experience, and ecosystem, and finally synthesize a recommendation."
)


class _RoleAwareProvider:
    """Returns planner JSON, per-step worker answers, then a synthesis."""

    def __init__(self) -> None:
        self.roles: list[str] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text = "\n".join(str(getattr(m, "content", m)) for m in (request.messages or []))
        lower = text.lower()
        # Order matters: synthesizer prompts embed worker text that may mention steps.
        if "you are a planner" in lower and "json array" in lower:
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
        if "results from the planned steps" in lower:
            self.roles.append("synthesizer")
            return ModelResponse(content="FINAL: Use Go for simple APIs, Rust for hot paths.")
        if "complete only this step" in lower:
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
        complexity_router=ComplexityRouter(),
    )
    built = agent.build()

    plan_calls = 0
    original = built._run_plan

    async def traced(agent_input, *, output_schema=None, ctx=None):
        nonlocal plan_calls
        plan_calls += 1
        return await original(agent_input, output_schema=output_schema, ctx=ctx)

    built._run_plan = traced  # type: ignore[method-assign]

    result = asyncio.run(built.arun(TOUGH_TASK))

    assert plan_calls == 1
    assert provider.roles.count("planner") == 1
    assert provider.roles.count("worker") == 4
    assert provider.roles.count("synthesizer") == 1
    assert "FINAL:" in result.output.text()
    # Planner first, workers fan out, synthesizer last.
    assert provider.roles[0] == "planner"
    assert provider.roles[-1] == "synthesizer"
    assert all(r == "worker" for r in provider.roles[1:-1])
