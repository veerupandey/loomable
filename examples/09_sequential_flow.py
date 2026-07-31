"""09 — Sequential Flow: Research → Draft → Edit

A 3-step pipeline where each step's output feeds into the next.
Agents are passed directly — the framework coerces outputs automatically.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import sequential
from loomable.providers.openai import AzureOpenAIProvider

# --- Create 3 agents with different instructions ---

provider = AzureOpenAIProvider()

researcher = Agent(
    model=provider,
    instructions="You are a researcher. List 3-5 key facts about the topic. Be concise, use bullet points.",
)

drafter = Agent(
    model=provider,
    instructions="You are a writer. Take the research notes and write a single coherent paragraph (3-4 sentences).",
)

editor = Agent(
    model=provider,
    instructions="You are an editor. Polish the draft for clarity, flow, and impact. Keep it to 2-3 sentences.",
)

# --- Compose agents directly into a sequential flow ---

pipeline = sequential(researcher, drafter, editor, session_id="article-pipeline")

result = asyncio.run(pipeline.arun("The history of the Python programming language"))
print("=== Final edited output ===")
print(result.output.text())
