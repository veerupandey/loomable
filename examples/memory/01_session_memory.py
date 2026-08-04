"""Session Memory — Multi-turn conversation within a session.

USE WHEN: You need the agent to remember what was said earlier
in the same conversation (e.g. a chat session).

session_id enables turn-by-turn memory that persists across
multiple arun() calls within the same process.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Personal Tutor",
    goal="Teach concepts building on previous explanations",
    instructions="Build on what you've already explained. Reference prior turns.",
    session_id="tutoring-session-001",
    memory_window=8,
)

# Turn 1
result = asyncio.run(agent.arun("Explain what a hash table is, for a beginner."))
print(f"Turn 1: {result.output.text()}\n")

# Turn 2: Agent remembers Turn 1
result = asyncio.run(agent.arun("Now explain how collisions are handled."))
print(f"Turn 2: {result.output.text()}")
