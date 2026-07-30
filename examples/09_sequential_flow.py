"""09 — Sequential Flow: Research → Draft → Edit

A 3-step pipeline where each step's output feeds into the next.
Uses the `sequential()` helper with function nodes wrapping agents.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import sequential
from loomable.providers.openai import AzureOpenAIProvider

# --- Create 3 agents with different instructions ---

provider = AzureOpenAIProvider()

research_agent = Agent(
    model=provider,
    instructions="You are a researcher. List 3-5 key facts about the topic. Be concise, use bullet points.",
)

draft_agent = Agent(
    model=provider,
    instructions="You are a writer. Take the research notes and write a single coherent paragraph (3-4 sentences).",
)

edit_agent = Agent(
    model=provider,
    instructions="You are an editor. Polish the draft for clarity, flow, and impact. Keep it to 2-3 sentences.",
)


# --- Wrap agents in functions so text flows between nodes ---


async def research(input, **kwargs):
    result = await research_agent.arun(str(input))
    return result.output.text()


async def draft(input, **kwargs):
    result = await draft_agent.arun(str(input))
    return result.output.text()


async def edit(input, **kwargs):
    result = await edit_agent.arun(str(input))
    return result.output.text()


# --- Compose into a sequential flow ---

pipeline = sequential(research, draft, edit, session_id="article-pipeline")

result = asyncio.run(pipeline.arun("The history of the Python programming language"))
print("=== Final edited output ===")
print(result.output.text())
