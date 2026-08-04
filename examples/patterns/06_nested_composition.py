"""Nested Composition — Flows containing other Flows.

USE WHEN: You have a complex workflow where individual stages
are themselves multi-step processes (composition of compositions).

Flow nodes can be Loops, other Flows, or any Runnable — they nest.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import sequential, parallel
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Stage 1: Research (parallel perspectives) ---

tech_researcher = Agent(
    model=provider,
    role="Technical Researcher",
    goal="Research technical feasibility",
    instructions="Assess technical feasibility in 2-3 sentences.",
)

market_researcher = Agent(
    model=provider,
    role="Market Researcher",
    goal="Research market viability",
    instructions="Assess market viability in 2-3 sentences.",
)

research_stage = parallel(tech_researcher, market_researcher)

# --- Stage 2: Analysis (sequential) ---

analyst = Agent(
    model=provider,
    role="Business Analyst",
    goal="Synthesize research into a recommendation",
    instructions="Given the research results, provide a clear go/no-go recommendation.",
)

# --- Full pipeline: parallel research → sequential analysis ---

pipeline = sequential(research_stage, analyst)

result = asyncio.run(pipeline.arun(
    "Should we build a real-time collaborative code editor as a SaaS product?"
))
print(result.output.text())
