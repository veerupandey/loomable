"""Parallel Execution — Parallel_Group within a Workflow

Shows how to run multiple steps concurrently using Parallel_Group.
Steps inside a Parallel_Group execute at the same time; the workflow
waits for all of them to finish before continuing.

Key concepts:
- Parallel_Group: wraps multiple Steps for concurrent execution
- Results from parallel steps are merged into sub_results
- Can be mixed with sequential Steps in the same Workflow
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import Step, Workflow, Parallel_Group
from loomable.providers.openai import AzureOpenAIProvider


# --- Setup: agents for different tasks ---

provider = AzureOpenAIProvider()

researcher = Agent(
    model=provider,
    instructions="You are a researcher. List 3-5 key facts about the topic. Be concise.",
)

translator = Agent(
    model=provider,
    instructions="Translate the text to French. Keep the same tone and style.",
)

summarizer = Agent(
    model=provider,
    instructions="Summarize the text in one sentence.",
)


# --- Workflow with parallel post-processing ---

workflow = Workflow(
    name="parallel_research",
    steps=[
        Step("research", researcher, description="Initial research"),
        Parallel_Group(
            Step("translate", translator),
            Step("summarize", summarizer),
            name="post_processing",
        ),
    ],
)

# Inspect the topology
plan = workflow.explain()
print(f"Topology: {plan.original_nodes}")

# Run — translate and summarize happen concurrently
result = asyncio.run(workflow.arun("Quantum computing fundamentals"))
print(f"\nParallel results available via sub_results")
print(f"Output:\n{result.output.text()}")
