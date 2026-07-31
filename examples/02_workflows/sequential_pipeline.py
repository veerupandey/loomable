"""Sequential Pipeline — Step + Workflow

Shows the most fundamental Workflow pattern: a series of Steps executed
in order, where each step's output feeds into the next. This is the
recommended way to build multi-agent pipelines in loomable.

Key concepts:
- Step: a named wrapper around an agent (or function)
- Workflow: compiles a steps list into an execution graph
- explain(): inspect the compiled topology before running
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import Step, Workflow
from loomable.providers.openai import AzureOpenAIProvider


# --- Setup: specialized agents for each pipeline stage ---

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


# --- Build the workflow ---

workflow = Workflow(
    name="article_pipeline",
    steps=[
        Step("research", researcher, description="Gather key facts"),
        Step("draft", drafter, description="Write a paragraph"),
        Step("edit", editor, description="Polish for clarity"),
    ],
)

# Inspect the compiled graph before running
plan = workflow.explain()
print(f"Compiled topology: {plan.original_nodes}")
print(f"Edges: {plan.original_edges}")

# Run the pipeline
result = asyncio.run(workflow.arun("The impact of AI on healthcare"))
print(f"\nFinal output:\n{result.output.text()}")
