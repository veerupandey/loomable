"""Nested composition — Workflows inside Workflows.

USE WHEN: Individual stages are themselves multi-step processes
(composition of compositions). Nested Workflows are first-class Runnables.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow

provider = require_provider()

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

analyst = Agent(
    model=provider,
    role="Business Analyst",
    goal="Synthesize research into a recommendation",
    instructions="Given the research results, provide a clear go/no-go recommendation.",
)

# Nested: research stage is its own Workflow, then analysis
research = Workflow("research").parallel(
    tech=tech_researcher,
    market=market_researcher,
)

pipeline = (
    Workflow("product-decision")
    .step("research", research)
    .step("analyze", analyst)
)

result = asyncio.run(
    pipeline.arun(
        "Should we build a real-time collaborative code editor as a SaaS product?"
    )
)
print(result.output.text())
