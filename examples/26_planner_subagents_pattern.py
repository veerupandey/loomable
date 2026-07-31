"""
High-level planner → subagents pattern

You do NOT parse agent output by hand.
You do NOT wire planner / worker / synthesizer yourself.

After Z.AI A/B experiments, the default ComplexityRouter is efficiency-biased:
it only auto-plans when signals are strong. To demo the plan→subagents path
reliably, this example forces PLAN via a tiny classifier.

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


class AlwaysPlan:
    """Force the plan → parallel subagents → synthesize path."""

    def classify(self, agent_input, *, has_tools: bool) -> RunStrategy:
        return RunStrategy.PLAN


TASK = (
    "Compare and analyze how to launch AI software that helps factories "
    "plan shop-floor work. Break down the work step by step. "
    "For each area cover: who to sell to, who we compete with, simple pricing, "
    "and a 90-day plan. Decompose into multiple steps, then synthesize "
    "one clear CEO answer in plain English."
)

provider = make_provider()
heuristic = ComplexityRouter()
forced = ComplexityRouter(model_classifier=AlwaysPlan())

agent = Agent(
    model=provider,
    instructions=(
        "Explain things in plain English. "
        "Use short sentences. Avoid jargon."
    ),
    # High-level API: framework owns plan → subagents → synthesize.
    complexity_router=forced,
)


async def main() -> None:
    heur = heuristic.classify(AgentInput.from_text(TASK), has_tools=False)
    print("High-level Agent API")
    print(f"  default heuristic would choose: {heur.value}")
    print("  this demo forces: plan  →  subagents  →  synthesize")
    print("-" * 60)
    print(f"Task: {TASK}\n")

    result = await agent.arun(TASK)

    print("-" * 60)
    print(f"run_strategy : {result.metadata.get('run_strategy')}")
    print(f"plan_workers : {result.metadata.get('plan_workers')}")
    print("-" * 60)
    print(result.output.text())
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
