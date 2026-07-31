"""
High-level planner → subagents pattern

You do NOT parse agent output by hand.
You do NOT wire planner / worker / synthesizer yourself.

Just give the Agent a complexity router (or plan_tool=True).
Loomable decides when to plan, creates subagents, passes results
between them, and returns one final answer.

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
from loomable.agent.routing import ComplexityRouter
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


# ---------------------------------------------------------------------------
# High-level API — one Agent, framework does the rest
# ---------------------------------------------------------------------------

agent = Agent(
    model=make_provider(),
    instructions=(
        "Explain things in plain English. "
        "Use short sentences. Avoid jargon."
    ),
    # When the task looks complex, Loomable auto-runs:
    #   plan → parallel subagents → synthesize
    # Outputs are passed inside the framework. No manual parsing.
    complexity_router=ComplexityRouter(),
    # Optional alternative: let the model call a `plan` tool itself.
    # plan_tool=True,
)

TASK = (
    "Help me launch AI software that helps factories plan shop-floor work. "
    "Cover who to sell to, who we compete with, simple pricing, "
    "and a 90-day plan. Keep it plain English."
)


async def main() -> None:
    print("High-level Agent API")
    print("  complexity_router=on  →  auto plan → subagents → final answer")
    print("-" * 60)
    print(f"Task: {TASK}\n")

    result = await agent.arun(TASK)

    print("-" * 60)
    print(result.output.text())
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
