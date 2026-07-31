"""
Planner → Subagents pattern (clean example)

Flow:
  1. Planner   — breaks one big job into small steps
  2. Workers   — each step runs as its own subagent (in parallel)
  3. Synthesize — combines worker answers into one final reply

Provider: any OpenAI-compatible API (Z.AI GLM shown below).

Setup (do NOT put secrets in this file):

  export ZAI_API_KEY="your-key"
  export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"   # optional
  export ZAI_MODEL="glm-5.2"                                  # optional

  python examples/26_planner_subagents_pattern.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from loomable.agent import Agent
from loomable.flow import plan_and_execute
from loomable.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Provider (OpenAI-compatible → Z.AI)
# ---------------------------------------------------------------------------

def make_provider() -> OpenAIProvider:
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ZAI_API_KEY (or OPENAI_API_KEY) in your environment.\n"
            "Example:\n"
            '  export ZAI_API_KEY="your-key"\n'
            "  python examples/26_planner_subagents_pattern.py"
        )
    return OpenAIProvider(
        model=os.environ.get("ZAI_MODEL", "glm-5.2"),
        api_key=api_key,
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
        timeout=180.0,
    )


PROVIDER = make_provider()


# ---------------------------------------------------------------------------
# 1) Planner — decides how many subagents / steps to create
# ---------------------------------------------------------------------------

planner_agent = Agent(
    model=PROVIDER,
    instructions=(
        "You are a planner. Reply in plain English thinking, but your final "
        "output must be ONLY a JSON object like:\n"
        '  {"plan_steps": ["step one", "step two", "step three"]}\n'
        "Make 3 to 5 short, independent steps. No markdown. No code fences."
    ),
)


async def planner(user_task, **kwargs):
    result = await planner_agent.arun(str(user_task))
    text = result.output.text().strip()

    # Prefer JSON object / array; fall back to bullet lines.
    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("plan_steps"), list):
            steps = [str(s).strip() for s in data["plan_steps"] if str(s).strip()]
            return {"plan_steps": steps[:5]}
        if isinstance(data, list):
            return {"plan_steps": [str(s).strip() for s in data if str(s).strip()][:5]}
    except (json.JSONDecodeError, ValueError):
        pass

    steps = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\s\-\*\u2022\d\.\)\(]+", "", line).strip()
        if cleaned:
            steps.append(cleaned)
    return {"plan_steps": steps[:5] or ["Do the whole task in one pass"]}


# ---------------------------------------------------------------------------
# 2) Worker subagent — runs ONE planned step
# ---------------------------------------------------------------------------

worker_agent = Agent(
    model=PROVIDER,
    instructions=(
        "You do one small job at a time. "
        "Answer in plain English with short sentences. "
        "Be concrete. No buzzwords."
    ),
)


async def worker(step, **kwargs):
    result = await worker_agent.arun(str(step))
    return result.output.text()


# ---------------------------------------------------------------------------
# 3) Synthesizer — merges all worker answers
# ---------------------------------------------------------------------------

synth_agent = Agent(
    model=PROVIDER,
    instructions=(
        "You write a clear final answer for a busy reader. "
        "Use plain English, short sentences, and simple headings."
    ),
)


async def synthesizer(state, **kwargs):
    # Map results arrive as a list (preferred) or dict{"map": [...]}.
    if isinstance(state, list):
        pieces = state
    elif isinstance(state, dict):
        pieces = state.get("map", []) or []
    else:
        pieces = []
    combined = "\n\n".join(f"Worker result:\n{p}" for p in pieces) if pieces else str(state)
    prompt = (
        f"Original task:\n{TASK}\n\n"
        "Combine these worker results into one final answer.\n"
        "Keep it simple and easy to read. Use short headings.\n\n"
        f"{combined}"
    )
    result = await synth_agent.arun(prompt)
    return result.output.text()


# ---------------------------------------------------------------------------
# Build the flow: planner → map(workers) → synthesizer
# ---------------------------------------------------------------------------

flow = plan_and_execute(
    planner=planner,
    workers=worker,
    synthesizer=synthesizer,
    session_id="planner-subagents-demo",
)


TASK = (
    "Help me launch AI software that helps factories plan shop-floor work. "
    "Cover: who to sell to, who we compete with, simple pricing, and a 90-day plan. "
    "Use plain English."
)


async def main() -> None:
    print("=" * 64)
    print("Pattern: Planner → Subagents → Final answer")
    print("=" * 64)
    print(f"model : {PROVIDER.model}")
    print(f"task  : {TASK}\n")

    # Show the plan first so the pattern is visible.
    plan = await planner(TASK)
    steps = plan["plan_steps"]
    print(f"Planner created {len(steps)} subagent steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    print("\nRunning full flow (workers in parallel)...\n")

    result = await flow.arun(TASK)

    print("-" * 64)
    print("Final answer")
    print("-" * 64)
    print(result.output.text())
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
