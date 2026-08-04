"""Plan and Execute — Dynamic task decomposition.

USE WHEN: The task is too complex for a single agent and the steps
aren't known in advance. The planner decomposes dynamically.

Uses plan_and_execute: a planner breaks the task into steps,
workers execute each step, a synthesizer combines results.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import plan_and_execute
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

planner = Agent(
    model=provider,
    role="Project Planner",
    goal="Decompose complex tasks into clear steps",
    instructions=(
        "Break the task into 2-4 concrete steps. "
        "Output a JSON list of step descriptions. "
        "Each step should be independently executable."
    ),
)

worker = Agent(
    model=provider,
    role="Task Worker",
    goal="Execute a single task step thoroughly",
    instructions="Complete the assigned step. Be thorough but concise.",
)

synthesizer = Agent(
    model=provider,
    role="Synthesizer",
    goal="Combine step results into a coherent final answer",
    instructions="Merge all step results into one cohesive response.",
)

flow = plan_and_execute(planner, worker, synthesizer)

result = asyncio.run(flow.arun(
    "Design a caching strategy for a social media feed API "
    "that handles 10K requests/second."
))
print(result.output.text())
