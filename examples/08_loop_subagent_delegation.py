"""08 — Loop with Subagent Delegation

A Loop where each iteration uses specialist tools (acting as subagents).
The main agent coordinates, calling different tools until the verifier passes.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Loop
from loomable.providers.openai import AzureOpenAIProvider


# --- Specialist "subagent" tools ---


@tool
def research(topic: str) -> str:
    """Research a topic and return factual findings."""
    # In production, this could call another agent or API
    findings = {
        "solar": "Solar panels: 20-22% efficiency, costs dropped 89% since 2010, 1000+ GW installed globally.",
        "wind": "Wind power: Onshore turbines 2-3 MW, offshore 12+ MW, 900+ GW installed globally.",
        "nuclear": "Nuclear: 90%+ capacity factor, 440 reactors worldwide, near-zero carbon emissions.",
    }
    for key, val in findings.items():
        if key in topic.lower():
            return val
    return f"Research on '{topic}': Renewable energy generates 30% of global electricity."


@tool
def fact_check(claim: str) -> str:
    """Verify a factual claim and return the verification result."""
    # Simulated fact-checking
    if any(num in claim for num in ["20", "22", "89", "90", "440", "1000"]):
        return "VERIFIED: Statistics confirmed by IEA/IRENA data."
    return "UNVERIFIED: Could not confirm. Please cite source."


# --- Verifier: answer must contain [VERIFIED] ---


def has_verification(output: AgentOutput, ctx: RunContext) -> bool:
    return "[VERIFIED]" in output.text().upper()


# --- Build and run ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a research assistant. Always:\n"
        "1. Use the research tool to gather facts\n"
        "2. Use the fact_check tool to verify claims\n"
        "3. Include [VERIFIED] in your final answer if facts check out\n"
        "Only include [VERIFIED] tag if fact_check confirms the data."
    ),
    tools=[research, fact_check],
)

loop = Loop(body=agent, verifier=has_verification, max_iterations=3)

result = asyncio.run(loop.arun("What is the current efficiency of solar panels?"))
print("Answer:", result.output.text())
print(f"\nVerified: {result.metadata.get('loop_verified')}")
print(f"Iterations: {result.metadata.get('loop_iterations')}")
