"""12 — Coordinate Flow: Workers + Manager Synthesis

Multiple worker agents run in parallel on the same task, then a manager
agent synthesizes all their outputs into one coherent response.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import coordinate
from loomable.providers.openai import AzureOpenAIProvider

# --- Worker agents (specialized reviewers) ---

provider = AzureOpenAIProvider()

security_agent = Agent(
    model=provider,
    instructions="You are a security reviewer. Identify security concerns in the described system. Be specific and concise (3-4 bullet points).",
)

performance_agent = Agent(
    model=provider,
    instructions="You are a performance engineer. Identify performance bottlenecks in the described system. Be specific and concise (3-4 bullet points).",
)

ux_agent = Agent(
    model=provider,
    instructions="You are a UX reviewer. Identify usability concerns in the described system. Be specific and concise (3-4 bullet points).",
)

# --- Manager agent (synthesizer) ---

manager_agent = Agent(
    model=provider,
    instructions=(
        "You are a tech lead. You receive reviews from security, performance, and UX experts. "
        "Synthesize their feedback into a prioritized action plan with the top 5 items to address first."
    ),
)


# --- Wrap agents in functions for flow compatibility ---


async def security(input, **kwargs):
    result = await security_agent.arun(str(input))
    return result.output.text()


async def performance(input, **kwargs):
    result = await performance_agent.arun(str(input))
    return result.output.text()


async def ux_reviewer(input, **kwargs):
    result = await ux_agent.arun(str(input))
    return result.output.text()


async def manager(input, **kwargs):
    result = await manager_agent.arun(str(input))
    return result.output.text()


# --- Coordinate: workers → manager ---

flow = coordinate(
    workers=[security, performance, ux_reviewer],
    manager=manager,
    session_id="system-review",
)

result = asyncio.run(flow.arun(
    "Review our e-commerce checkout system: React frontend, Node.js API, "
    "PostgreSQL database, Stripe payments, session-based auth, no caching layer."
))
print("=== Synthesized Review ===")
print(result.output.text())
