"""Flow Shared Memory — Agents in a flow share state.

USE WHEN: Multiple agents in a pipeline need to read/write
shared state that persists across the flow execution.

The TieredMemoryStore (memory parameter) is shared across
all nodes in a Flow, enabling inter-agent communication.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import sequential
from loomable.flow.memory import TieredMemoryStore
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# Shared memory across the flow
shared_memory = TieredMemoryStore()

researcher = Agent(
    model=provider,
    role="Researcher",
    goal="Research and store findings",
    instructions="Research the topic. Store key findings clearly.",
)

writer = Agent(
    model=provider,
    role="Writer",
    goal="Write based on stored research",
    instructions="Write a summary based on what the researcher found.",
)

pipeline = sequential(researcher, writer, memory=shared_memory)

result = asyncio.run(pipeline.arun("What are the pros and cons of event sourcing?"))
print(result.output.text())
