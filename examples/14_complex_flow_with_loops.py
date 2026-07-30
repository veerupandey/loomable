"""14 — Complex Flow with Loop Nodes

Demonstrates composability: Flow nodes that are themselves Loops.
A sequential flow where the "draft" step is a Loop that retries until
the output passes quality checks.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Loop, sequential
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Step 1: Research (simple agent) ---

researcher_agent = Agent(
    model=provider,
    instructions="Research the topic. List 4-5 key facts as bullet points.",
)


# --- Step 2: Draft (Loop — retries until quality passes) ---

drafter_agent = Agent(
    model=provider,
    instructions=(
        "Write a single paragraph (exactly 3 sentences) summarizing the key facts. "
        "Include at least one specific number or statistic. "
        "Do NOT use bullet points — write flowing prose."
    ),
)


def quality_check(output: AgentOutput, ctx: RunContext) -> bool:
    """Draft must be 2-4 sentences, contain a number, and not use bullet points."""
    text = output.text().strip()
    sentences = [s.strip() for s in text.replace("...", ".").split(".") if s.strip()]
    has_right_length = 2 <= len(sentences) <= 4
    has_numbers = any(c.isdigit() for c in text)
    no_bullets = not any(text.lstrip().startswith(c) for c in "-*•")
    return has_right_length and has_numbers and no_bullets


# --- Step 3: Polish (simple agent) ---

polisher_agent = Agent(
    model=provider,
    instructions="Polish this text for clarity and elegance. Keep it concise (2-3 sentences max).",
)


# --- Wrap agents in functions for flow compatibility ---


async def researcher(input, **kwargs):
    result = await researcher_agent.arun(str(input))
    return result.output.text()


async def drafter(input, **kwargs):
    result = await drafter_agent.arun(str(input))
    return result.output.text()


async def polisher(input, **kwargs):
    result = await polisher_agent.arun(str(input))
    return result.output.text()


# --- Build loop with drafter function ---

draft_loop = Loop(body=drafter, verifier=quality_check, max_iterations=3)


# --- Compose: research → draft_loop → polish ---

pipeline = sequential(researcher, draft_loop, polisher)

print("Running: research → draft(loop) → polish\n")
result = asyncio.run(pipeline.arun("The impact of artificial intelligence on healthcare"))
print("=== Final Output ===")
print(result.output.text())
