"""Pipeline — Sequential chain of agents.

USE WHEN: You have a multi-step workflow where each step's
output feeds into the next step's input.

Uses the `sequential` flow helper. Each Agent receives the previous
Agent's output automatically — no parse/glue functions between steps.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import sequential
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

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
