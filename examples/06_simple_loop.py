"""06 — Simple Loop with Verifier

A Loop repeats its body until a Verifier passes or max iterations are reached.
This example: generate a haiku, verify it has exactly 3 lines.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Loop
from loomable.providers.openai import AzureOpenAIProvider


# --- Verifier: check that output has exactly 3 non-empty lines ---


def haiku_verifier(output: AgentOutput, ctx: RunContext) -> bool:
    lines = [l for l in output.text().strip().split("\n") if l.strip()]
    return len(lines) == 3


# --- Build the loop ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "Write a haiku (exactly 3 lines, 5-7-5 syllable pattern). "
        "Output ONLY the 3 lines of the haiku, nothing else."
    ),
)

loop = Loop(
    body=agent,
    verifier=haiku_verifier,
    max_iterations=3,
)

result = asyncio.run(loop.arun("Write a haiku about programming."))
print("Final haiku:")
print(result.output.text())
print(f"\nIterations used: {result.metadata.get('loop_iterations', '?')}")
print(f"Verified: {result.metadata.get('loop_verified', '?')}")
