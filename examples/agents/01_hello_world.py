"""Hello World Agent — The absolute minimum.

USE WHEN: You just want a single agent to answer a question.
This is the starting point for any loomable project.

One agent, one question, one answer — 3 lines of setup.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Helpful Assistant",
    goal="Answer questions clearly and concisely",
)

result = asyncio.run(agent.arun("What is the capital of France? Answer in one sentence."))
print(result.output.text())
