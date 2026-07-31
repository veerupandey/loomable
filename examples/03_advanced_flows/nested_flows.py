"""15 — Nested Flows with Agent Nodes

A complex flow with multiple levels of composition:
- Top-level: sequential(analyze → coordinate_review)
- Middle level: coordinate has 3 worker agents running in parallel + manager
- Each worker is a full agent with different expertise

This demonstrates the full composability of the framework.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import sequential, coordinate
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Analyzer agent (first step) ---

analyzer_agent = Agent(
    model=provider,
    instructions="Analyze the system described and produce a brief technical summary (architecture, stack, key components).",
)

# --- Review workers (run in parallel) ---

api_reviewer_agent = Agent(
    model=provider,
    instructions="Review the API design aspects. List 3 recommendations for improvement.",
)

data_reviewer_agent = Agent(
    model=provider,
    instructions="Review the data model and storage aspects. List 3 recommendations.",
)

infra_reviewer_agent = Agent(
    model=provider,
    instructions="Review the infrastructure and deployment aspects. List 3 recommendations.",
)

# --- Manager (synthesizes reviews) ---

manager_agent = Agent(
    model=provider,
    instructions=(
        "You are a principal engineer. Synthesize the API, data, and infrastructure reviews "
        "into a prioritized improvement roadmap (top 5 items, ordered by impact)."
    ),
)


# --- Compose: analyze → coordinate(workers + manager) ---

review_flow = coordinate(
    workers=[api_reviewer_agent, data_reviewer_agent, infra_reviewer_agent],
    manager=manager_agent,
)

full_pipeline = sequential(analyzer_agent, review_flow, session_id="nested-review")

print("Running nested flow: analyze → coordinate(3 workers + manager)\n")
result = asyncio.run(full_pipeline.arun(
    "System: A social media API built with Express.js, MongoDB, Redis cache, "
    "deployed on EC2 with manual scaling, using JWT auth and S3 for media storage."
))
print("=== Final Improvement Roadmap ===")
print(result.output.text())
