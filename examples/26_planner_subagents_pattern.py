"""
High-level planner → subagents pattern

Best API:

  Agent(model=..., plan="always")   # force plan → parallel subagents → synthesize
  Agent(model=..., plan=True)       # auto (ComplexityRouter)
  Agent(model=..., plan="never")    # never auto-plan (default)

No hand parsing. Framework passes outputs between stages.

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


TASK = (
    "Compare and analyze how to launch AI software that helps factories "
    "plan shop-floor work. Break down the work step by step. "
    "For each area cover: who to sell to, who we compete with, simple pricing, "
    "and a 90-day plan. Decompose into multiple steps, then synthesize "
    "one clear CEO answer in plain English."
)

agent = Agent(
    model=make_provider(),
    instructions="Explain in plain English. Short sentences. Avoid jargon.",
    # High-level: framework owns plan → subagents → synthesize.
    plan="always",
)


async def main() -> None:
    print("High-level Agent API  —  plan=\"always\"")
    print("  framework: plan → parallel subagents → synthesize")
    print("-" * 60)
    print(f"Task: {TASK}\n")

    result = await agent.arun(TASK)

    print("-" * 60)
    print(f"run_strategy : {result.metadata.get('run_strategy')}")
    print(f"plan_trigger : {result.metadata.get('plan_trigger')}")
    print(f"plan_workers : {result.metadata.get('plan_workers')}")
    print(f"plan_steps   : {result.metadata.get('plan_steps')}")
    print("-" * 60)
    print(result.output.text())
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
