"""25 — Tough use case: does the agent auto-plan and fan out?

Assigns a large multi-part product strategy task to a Loomable Agent with
ComplexityRouter enabled. Demonstrates:

1. Router auto-selects PLAN for a complex prompt (no manual mode choice)
2. Planner decides how many steps/workers to create at runtime
3. Workers fan out via plan_and_execute MapNode
4. Synthesizer merges results into a final answer

Uses a prompt-aware FakeProvider (no API key required) that behaves like an
LLM across planner / worker / synthesizer roles so the *framework path* can
be observed end-to-end. Swap in AzureOpenAIProvider / OpenAIProvider to try
with a real model.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loomable.agent import Agent, ModelSpec
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.kernel.models import ModelRequest, ModelResponse


TOUGH_TASK = """
Design a go-to-market plan for a new multi-tenant B2B SaaS product that helps
mid-market manufacturers modernize shop-floor scheduling with AI.

Compare and analyze, then break down the work step by step. Cover ALL of:
1) ICP and buyer personas (plant manager, COO, IT)
2) Competitive landscape vs traditional MES / APS vendors
3) Pricing packaging for SMB vs mid-market
4) Security / compliance requirements (SOC2, data residency)
5) 90-day launch plan with milestones, owners, and risks
6) Success metrics and experiment backlog for the first two quarters

For each area first research constraints, then synthesize recommendations,
and finally produce an executive brief a CEO could act on.
""".strip()


class TracingPlanProvider:
    """Fake LLM that responds differently for planner / worker / synthesizer prompts."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text = self._messages_text(request)
        role = self._classify_role(text)
        self.calls.append({"role": role, "preview": text[:180].replace("\n", " ")})

        if role == "planner":
            steps = [
                "Define ICP and buyer personas for plant managers, COOs, and IT",
                "Map competitive landscape against MES and APS vendors",
                "Design pricing packaging for SMB vs mid-market segments",
                "Enumerate SOC2 and data-residency security requirements",
                "Draft a 90-day launch plan with milestones, owners, and risks",
            ]
            return ModelResponse(content=json.dumps(steps))

        if role == "worker":
            step = self._extract_step(text)
            return ModelResponse(
                content=(
                    f"Worker result for: {step}\n"
                    "- Key findings drafted with concrete recommendations.\n"
                    "- Risks called out; ready for synthesis."
                )
            )

        if role == "synthesizer":
            return ModelResponse(
                content=(
                    "# Executive Brief: AI Shop-Floor Scheduling GTM\n\n"
                    "ICP: Mid-market manufacturers (200–2000 employees) with aging MES.\n"
                    "Positioning: Faster schedule recovery than APS, lighter than full MES rip/replace.\n"
                    "Pricing: SMB starter + mid-market platform with usage-based optimization add-on.\n"
                    "Compliance: SOC2 Type II path + region-pinned tenants.\n"
                    "90-day plan: design partners → security review → pilot → priced GA.\n"
                    "KPIs: design-partner win rate, schedule OTIF lift, paid pilot conversion."
                )
            )

        return ModelResponse(content="Direct answer (unexpected path).")

    @staticmethod
    def _messages_text(request: ModelRequest) -> str:
        chunks: list[str] = []
        for msg in request.messages or []:
            content = getattr(msg, "content", msg)
            chunks.append(str(content))
        return "\n".join(chunks)

    @staticmethod
    def _classify_role(text: str) -> str:
        lower = text.lower()
        if "you are a planner" in lower and "json array" in lower:
            return "planner"
        if "complete only this step" in lower:
            return "worker"
        if "integrate these into one cohesive" in lower or "results from the planned steps" in lower:
            return "synthesizer"
        return "other"

    @staticmethod
    def _extract_step(text: str) -> str:
        match = re.search(
            r"Complete ONLY this step, concisely and concretely:\n(.+)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else "unknown step"


async def main() -> None:
    print("=" * 72)
    print("TOUGH USE CASE — Loomable auto-plan demo")
    print("=" * 72)
    print("\nTask assigned to agent:\n")
    print(TOUGH_TASK)
    print("\n" + "-" * 72)

    router = ComplexityRouter()
    strategy = router.classify(AgentInput.from_text(TOUGH_TASK), has_tools=False)
    print(f"\n1) ComplexityRouter decision: {strategy.value}")
    assert strategy == RunStrategy.PLAN, f"Expected PLAN, got {strategy}"
    print("   ✓ Auto-escalated to PLAN (no manual mode selection)")

    provider = TracingPlanProvider()
    agent = Agent(
        model=ModelSpec(provider="trace-fake", provider_impl=provider),
        instructions=(
            "You are a principal product strategist. For complex multi-part "
            "requests, rely on the framework's planning path."
        ),
        complexity_router=router,
    )
    built = agent.build()

    # Instrument _run_plan so we can prove auto-invocation.
    plan_invocations = 0
    original_run_plan = built._run_plan

    async def _traced_run_plan(agent_input, *, output_schema=None, ctx=None):
        nonlocal plan_invocations
        plan_invocations += 1
        print("\n2) BuiltAgent._run_plan() invoked automatically")
        return await original_run_plan(agent_input, output_schema=output_schema, ctx=ctx)

    built._run_plan = _traced_run_plan  # type: ignore[method-assign]

    print("\n3) Running agent.arun(...) — watching planner → workers → synthesizer\n")
    result = await built.arun(TOUGH_TASK)

    roles = [c["role"] for c in provider.calls]
    planner_calls = roles.count("planner")
    worker_calls = roles.count("worker")
    synth_calls = roles.count("synthesizer")

    print("   Model call trace:")
    for i, call in enumerate(provider.calls, 1):
        print(f"     [{i}] role={call['role']:<12} {call['preview'][:90]}...")

    print("\n4) Fan-out summary")
    print(f"   planner calls     : {planner_calls}")
    print(f"   worker calls      : {worker_calls}  ← dynamic subagent steps")
    print(f"   synthesizer calls : {synth_calls}")
    print(f"   _run_plan count   : {plan_invocations}")

    assert plan_invocations == 1
    assert planner_calls == 1
    assert worker_calls >= 2, "Planner should have created multiple worker steps"
    assert synth_calls == 1

    print("\n5) Final synthesized answer:\n")
    print(result.output.text())
    print("\n" + "=" * 72)
    print(
        f"RESULT: YES — tough task auto-escalated to PLAN, planner created "
        f"{worker_calls} workers on the fly, then synthesized."
    )
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
