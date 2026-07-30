"""16 — Agent Conversational Memory

Demonstrates conversational memory: the agent remembers previous turns.
Each call to `arun` records the exchange in session history, and subsequent
calls see prior turns in context.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

# --- Build agent with memory enabled ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a friendly assistant with a great memory. "
        "Always reference previous conversation turns when relevant."
    ),
    session_id="memory-demo",  # Enables session persistence + memory
    memory_window=8,  # Keep last 8 turns in context
    compaction_threshold=16,  # Summarize when turns exceed 16
)
built = agent.build()

# --- Multi-turn conversation ---

conversations = [
    "My name is Alex and I'm building a Python web app.",
    "What framework would you recommend for my project?",
    "What's my name and what am I building?",
]

for msg in conversations:
    result = asyncio.run(built.arun(msg))
    print(f"User: {msg}")
    print(f"Agent: {result.output.text()}")
    print()

# Show session state
print("--- Session state ---")
print(f"L1 turns (raw history): {len(built.session.l1)}")
print(f"L2 summaries (compacted): {len(built.session.l2)}")
print(f"Current step: {built.session.step}")
