"""Nested Workflows — Workflow inside Workflow

Shows how to compose pipelines by nesting one Workflow inside another.
A Workflow is itself a valid step, so you can build complex multi-stage
systems from smaller, testable sub-pipelines.

Key concepts:
- A Workflow can appear directly in another Workflow's steps list
- The inner workflow is treated as a single atomic step
- explain() shows the full nested topology
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import Step, Workflow
from loomable.providers.openai import AzureOpenAIProvider


# --- Setup ---

provider = AzureOpenAIProvider()

researcher = Agent(
    model=provider,
    instructions="You are a researcher. List 3-5 key facts about the topic. Be concise.",
)

drafter = Agent(
    model=provider,
    instructions="Take research notes and write a single coherent paragraph (3-4 sentences).",
)

editor = Agent(
    model=provider,
    instructions="Polish the draft for clarity and impact. Keep it to 2-3 sentences.",
)

translator = Agent(
    model=provider,
    instructions="Translate the text to French. Keep the same tone and style.",
)


# --- Inner workflow: research + draft ---

research_pipeline = Workflow(
    name="research_and_draft",
    steps=[
        Step("research", researcher),
        Step("draft", drafter),
    ],
)

# --- Outer workflow: uses inner pipeline as a step, then edits + translates ---

full_pipeline = Workflow(
    name="full_content_pipeline",
    steps=[
        research_pipeline,  # Nested workflow treated as a single step
        Step("edit", editor),
        Step("translate", translator),
    ],
)

# Inspect the full nested topology
plan = full_pipeline.explain()
print(f"Nested topology: {plan.original_nodes}")

# Run
result = asyncio.run(full_pipeline.arun("Benefits of renewable energy"))
print(f"\nFinal translated output:\n{result.output.text()}")
