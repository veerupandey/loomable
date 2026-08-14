"""Sequential Workflow — chain of Agents.

USE WHEN: Multi-step work where each Agent's output feeds the next.

Each Agent receives the previous Agent's output automatically —
no parse/glue functions between steps.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow
from loomable.display import pp, show_graph, step_outputs

provider = require_provider()

researcher = Agent(
    model=provider,
    role="Researcher",
    goal="Gather key facts about a topic",
    instructions="Output bullet points of key facts. Be thorough but concise.",
)

writer = Agent(
    model=provider,
    role="Technical Writer",
    goal="Turn research into polished prose",
    instructions="Take the bullet points and write a clear, flowing paragraph.",
)

editor = Agent(
    model=provider,
    role="Editor",
    goal="Polish and tighten the writing",
    instructions="Improve clarity, fix any issues, keep it under 150 words.",
)

pipeline = (
    Workflow("research-write-edit")
    .step("research", researcher)
    .step("write", writer)
    .step("edit", editor)
)

result = asyncio.run(pipeline.arun("Explain how garbage collection works in Python."))

pp(result)

steps = step_outputs(result)
for name, text in steps.items():
    print(f"\n--- Step: {name} ---")
    print(text[:200])

print()
show_graph(pipeline.flow, title="Research → Write → Edit")
