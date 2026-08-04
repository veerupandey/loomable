"""Subagents with Memory Sharing — Parent context flows to children.

USE WHEN: Your subagents need access to the same conversation
history or shared context as the parent.

Subagents inherit the parent's session context by default,
so they can reference what was discussed earlier.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

researcher = Agent(
    model=provider,
    role="Researcher",
    goal="Find accurate technical information",
    instructions="Provide factual, detailed research results.",
    session_id="shared-context",
)

writer = Agent(
    model=provider,
    role="Technical Writer",
    goal="Write clear documentation from research",
    instructions="Write concise, well-structured documentation.",
    session_id="shared-context",
)

lead = Agent(
    model=provider,
    role="Documentation Lead",
    goal="Produce high-quality technical documentation",
    instructions=(
        "First delegate research, then delegate writing based on the research. "
        "Return the final documentation."
    ),
    subagents=[researcher, writer],
    session_id="shared-context",
)

result = asyncio.run(lead.arun(
    "Create documentation for a REST API rate limiter middleware."
))
print(result.output.text())
