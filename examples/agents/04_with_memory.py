"""Agent with Conversational Memory — Multi-turn remembering.

USE WHEN: You need the agent to remember context from earlier
in the conversation (e.g. chatbots, multi-step workflows).

The session_id enables memory persistence across calls.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Friendly Assistant",
    goal="Maintain context across a multi-turn conversation",
    instructions="Always reference what the user told you earlier.",
    session_id="memory-demo-session",
    memory_window=8,
)

# Turn 1: Tell the agent something
result = asyncio.run(agent.arun("My name is Alex and I prefer concise answers."))
print(f"User: My name is Alex and I prefer concise answers.")
print(f"Agent: {result.output.text()}\n")

# Turn 2: The agent remembers
result = asyncio.run(agent.arun("What's my name and how should you respond to me?"))
print(f"User: What's my name and how should you respond to me?")
print(f"Agent: {result.output.text()}")
