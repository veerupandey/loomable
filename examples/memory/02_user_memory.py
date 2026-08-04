"""User Memory with Postgres — Cross-session persistence.

USE WHEN: You need the agent to remember user preferences and
facts across completely separate sessions (restarts, deployments).

Requires: pip install asyncpg
Requires: A running PostgreSQL instance.

The user_id scopes memory per user so the same agent serves
multiple users with isolated long-term memory.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

# NOTE: This example requires a PostgreSQL database.
# Uncomment and configure the connection URL for your environment.
#
# from loomable.providers.backends.postgres import PostgresMemoryBackend
#
# memory_backend = PostgresMemoryBackend(
#     url="postgresql://user:password@localhost/agentdb",
#     user_id="alice",
# )

provider = AzureOpenAIProvider()

# With Postgres backend (production):
# agent = Agent(
#     model=provider,
#     role="Personal Assistant",
#     goal="Remember user preferences across sessions",
#     session_id="session-abc",
#     user_id="alice",
#     memory=memory_backend,
# )

# Demo without Postgres (in-memory):
agent = Agent(
    model=provider,
    role="Personal Assistant",
    goal="Remember user preferences across sessions",
    instructions="Remember everything the user tells you about their preferences.",
    session_id="demo-session",
    user_id="alice",
)

# Session 1: User tells the agent something
result = asyncio.run(agent.arun("I prefer dark mode and Python over JavaScript."))
print(f"Session 1: {result.output.text()}\n")

# Session 2 (simulated): Agent recalls
result = asyncio.run(agent.arun("What are my preferences?"))
print(f"Session 2: {result.output.text()}")
