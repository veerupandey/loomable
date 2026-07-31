"""Conditional Branching — Condition in a Workflow

Shows declarative if/else branching within a Workflow. A Condition node
evaluates a function against the current SharedState and routes execution
to either then_steps or else_steps.

Key concepts:
- Condition: declarative if/else based on a predicate function
- then_steps / else_steps: branches containing Steps
- SharedState: carries data between steps for condition evaluation
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import Step, Workflow, Condition
from loomable.flow.state import SharedState
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

detailed_researcher = Agent(
    model=provider,
    instructions="Expand on the research. Add 3 more facts with specific numbers.",
)


# --- Condition function ---


def needs_more_detail(state: SharedState) -> bool:
    """Check if the research output is too short and needs expansion."""
    # In a real scenario, check SharedState for content length
    return state.get("needs_expansion", False)


# --- Workflow with conditional branching ---

workflow = Workflow(
    name="smart_pipeline",
    steps=[
        Step("research", researcher),
        Condition(
            condition=needs_more_detail,
            then_steps=[Step("expand", detailed_researcher)],
            else_steps=[Step("draft", drafter)],
        ),
        Step("edit", editor),
    ],
)

# Inspect — note the branching in the topology
plan = workflow.explain()
print(f"Topology includes branching: {len(plan.original_nodes)} nodes")

# Run
result = asyncio.run(workflow.arun("History of Python"))
print(f"\nFinal output:\n{result.output.text()}")
