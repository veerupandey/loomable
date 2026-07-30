"""07 — Loop with Tool-Using Agent

The body of a Loop is an agent with tools. Each iteration, the agent can call
tools to check its work and refine based on verifier feedback.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Loop
from loomable.providers.openai import AzureOpenAIProvider


# --- Tools ---


@tool
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


@tool
def char_count(text: str) -> int:
    """Count the number of characters in a text."""
    return len(text)


# --- Verifier: answer must be between 20 and 40 words ---


def length_check(output: AgentOutput, ctx: RunContext) -> bool:
    words = len(output.text().split())
    return 20 <= words <= 40


# --- Build and run ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "Summarize topics. Your summary MUST be between 20 and 40 words. "
        "Use the word_count tool to check your answer length before finalizing. "
        "If too short or too long, adjust."
    ),
    tools=[word_count, char_count],
)

loop = Loop(body=agent, verifier=length_check, max_iterations=3)

result = asyncio.run(loop.arun("Explain the theory of relativity in 20-40 words."))
print("Summary:", result.output.text())
print(f"Word count: {len(result.output.text().split())}")
print(f"Iterations: {result.metadata.get('loop_iterations')}")
print(f"Verified: {result.metadata.get('loop_verified')}")
