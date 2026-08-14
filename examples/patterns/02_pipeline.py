"""Pipeline — Sequential chain of agents.

USE WHEN: You have a multi-step workflow where each step's
output feeds into the next step's input.

Uses the `sequential` flow helper. Each Agent receives the previous
Agent's output automatically — no parse/glue functions between steps.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent
from loomable.flow.helpers import sequential

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

pipeline = sequential(researcher, writer, editor)

result = asyncio.run(pipeline.arun("Explain how garbage collection works in Python."))

# Pretty-print with per-step breakdown
from loomable.display import pp, step_outputs, show_graph

pp(result)

# Access individual step outputs by node name
steps = step_outputs(result)
for name, text in steps.items():
    print(f"\n--- Step: {name} ---")
    print(text[:200])

# Visualize the flow graph (Mermaid syntax — paste into mermaid.live)
print()
show_graph(pipeline, title="Research → Write → Edit")
