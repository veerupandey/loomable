"""Checkpointing — Durable state and HITL resume.

USE WHEN: Your workflow is long-running and you need to pause/resume
it across process restarts, or you have human-in-the-loop approval
gates that may take hours.

Checkpoints persist flow state so execution can resume exactly
where it left off.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import sequential
from loomable.persist.checkpoint import JsonFileCheckpointer
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# Steps in a durable pipeline
draft_agent = Agent(
    model=provider,
    role="Drafter",
    goal="Write an initial draft",
    instructions="Write a short draft (2-3 sentences) on the topic.",
)

review_agent = Agent(
    model=provider,
    role="Reviewer",
    goal="Review and approve or request changes",
    instructions="Review the draft. Approve if good, suggest changes if not.",
)

pipeline = sequential(
    draft_agent,
    review_agent,
    session_id="checkpoint-demo",
)

# In production, you'd configure a persistent checkpoint store:
# checkpointer = JsonFileCheckpointer(".checkpoints")
# Then pass: sequential(..., checkpointer=checkpointer)

result = asyncio.run(pipeline.arun("Explain why testing is important."))
print(result.output.text())
print("\nNote: In production, checkpoints allow resume after process restart.")
