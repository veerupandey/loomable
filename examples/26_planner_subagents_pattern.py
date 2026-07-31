"""
High-level planner → subagents pattern

You do NOT parse agent output by hand.
You do NOT wire planner / worker / synthesizer yourself.

  Agent + ComplexityRouter
    → framework plans
    → runs subagents
    → passes results between them
    → returns one final answer

Setup (secrets stay in the environment — never in this file):

  export ZAI_API_KEY="your-key"
  export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"   # optional
  export ZAI_MODEL="glm-5.2"                                  # optional

  python examples/26_planner_subagents_pattern.py
"""

from __future__ import annotations

import asyncio
import os

from loomable.agent import Agent
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.providers.openai import OpenAIProvider


def make_provider() -> OpenAIProvider:
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ZAI_API_KEY (or OPENAI_API_KEY), then re-run.\n"
            '  export ZAI_API_KEY="your-key"\n'
            "  python examples/26_planner_subagents_pattern.py"
        )
    return OpenAIProvider(
        model=os.environ.get("ZAI_MODEL", "glm-5.2"),
        api_key=api_key,
        base_url=os.environ.get(
            "ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"
        ),
        timeout=180.0,
    )


# Task wording matters: ComplexityRouter looks for cues like
# "compare", "step by step", "for each", "break down", "decompose".
TASK = (
    "Compare and analyze how to launch AI software that helps factories "
    "plan shop-floor work. Break down the work step by step. "
    "For each area cover: who to sell to, who we compete with, simple pricing, "
    "and a 90-day plan. Decompose into multiple steps, then synthesize "
    "one clear CEO answer in plain English."
)


# ---------------------------------------------------------------------------
# High-level API — one Agent, framework does the rest
# ---------------------------------------------------------------------------

router = ComplexityRouter()

agent = Agent(
    model=make_provider(),
    instructions=(
        "Explain things in plain English. "
        "Use short sentences. Avoid jargon."
    ),
    # When this returns PLAN, Loomable runs:
    #   plan → parallel subagents → synthesize
    # and passes outputs between them for you.
    complexity_router=router,
    # Optional: plan_tool=True  # model can call `plan` itself
)


async def main() -> None:
    strategy = router.classify(AgentInput.from_text(TASK), has_tools=False)
    print("High-level Agent API")
    print(f"  ComplexityRouter chose: {strategy.value}")
    if strategy != RunStrategy.PLAN:
        print("  Expected PLAN for this demo task — check task cues.")
        raise SystemExit(1)
    print("  framework will: plan → subagents → synthesize")
    print("-" * 60)
    print(f"Task: {TASK}\n")

    result = await agent.arun(TASK)

    print("-" * 60)
    print(result.output.text())
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
