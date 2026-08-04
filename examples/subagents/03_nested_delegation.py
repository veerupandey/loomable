"""Nested Delegation — A subagent has its own subagents.

USE WHEN: Your problem has multiple layers of decomposition,
where a specialist needs their own team of helpers.

Subagents can themselves have subagents, forming a delegation tree.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# Level 2: The research team (subagents of the researcher)
api_researcher = Agent(
    model=provider,
    role="API Researcher",
    goal="Research API design patterns and best practices",
)

security_researcher = Agent(
    model=provider,
    role="Security Researcher",
    goal="Research security best practices for the topic",
)

# Level 1: The main researcher has its own subagents
researcher = Agent(
    model=provider,
    role="Lead Researcher",
    goal="Coordinate research across API and security domains",
    instructions="Delegate to your specialists, then synthesize findings.",
    subagents=[api_researcher, security_researcher],
)

# Level 1: Writer (no subagents)
writer = Agent(
    model=provider,
    role="Technical Writer",
    goal="Write clear, actionable documentation",
)

# Level 0: Top-level coordinator
lead = Agent(
    model=provider,
    role="Project Lead",
    goal="Deliver comprehensive technical guides",
    instructions="Get research first, then have the writer produce the final doc.",
    subagents=[researcher, writer],
)

result = asyncio.run(lead.arun(
    "Write a guide on implementing OAuth 2.0 for a REST API."
))
print(result.output.text())
