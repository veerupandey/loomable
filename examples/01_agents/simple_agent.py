"""01 — Simple Agent (3 lines)

The absolute minimum: build an agent, run it, print the output.
Uses your Azure OpenAI deployment.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

# --- Build and run with real Azure OpenAI ---

provider = AzureOpenAIProvider()  # Reads from .env automatically

agent = Agent(model=provider, instructions="You are a helpful assistant. Be concise.")

result = asyncio.run(agent.arun("What is the capital of France? Answer in one sentence."))
print(result.output.text())
