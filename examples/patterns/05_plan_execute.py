"""Plan and Execute — dynamic task decomposition via Workflow.map.

USE WHEN: The task is too complex for a single agent and the steps
aren't known in advance. A planner decomposes; workers execute; a
synthesizer merges.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow

provider = require_provider()

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

wf = Workflow("plan-execute").map(
    worker,
    planner=planner,
    synthesizer=synthesizer,
)

result = asyncio.run(
    wf.arun(
        "Design a caching strategy for a social media feed API "
        "that handles 10K requests/second."
    )
)
print(result.output.text())
