"""13 — Plan and Execute Flow: Dynamic Task Decomposition

A planner generates steps, a MapNode fans them out to workers concurrently,
and a synthesizer combines the results. Uses real LLM for all three stages.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import plan_and_execute
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Planner: breaks a task into steps (returns dict with plan_steps list) ---

planner_agent = Agent(
    model=provider,
    instructions=(
        "You are a project planner. Break the user's task into 3-4 concrete, independent steps. "
        "Return ONLY a JSON object like: {\"plan_steps\": [\"step1\", \"step2\", ...]}. "
        "No markdown, no code fences, just the JSON."
    ),
)


async def planner(input, **kwargs):
    import json
    result = await planner_agent.arun(str(input))
    text = result.output.text().strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: split by lines
        steps = [l.strip().lstrip("-*•0123456789.) ") for l in text.splitlines() if l.strip()]
        return {"plan_steps": steps[:4]}


# --- Worker: executes each step ---

worker_agent = Agent(
    model=provider,
    instructions="Complete the given task step concisely in 2-3 sentences. Be specific and actionable.",
)


async def worker(input, **kwargs):
    result = await worker_agent.arun(str(input))
    return result.output.text()


# --- Synthesizer: combines all results ---

synth_agent = Agent(
    model=provider,
    instructions="Combine the step results into a cohesive final answer. Be well-structured and comprehensive.",
)


async def synthesizer(input, **kwargs):
    results = input.get("map", []) if isinstance(input, dict) else [str(input)]
    combined = "\n".join(f"- {r}" for r in results)
    result = await synth_agent.arun(f"Combine these results into a final answer:\n{combined}")
    return result.output.text()


# --- Build and run ---

flow = plan_and_execute(
    planner=planner,
    workers=worker,
    synthesizer=synthesizer,
    session_id="project-plan",
)

print("Running plan-and-execute flow...\n")
result = asyncio.run(flow.arun("Design a REST API for a todo-list application"))
print("=== Final Result ===")
print(result.output.text())
