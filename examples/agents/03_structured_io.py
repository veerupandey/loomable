"""Structured I/O — Validated input and output schemas.

USE WHEN: You need the agent to return structured data (JSON)
that conforms to a specific schema, not free-form text.

Uses Pydantic models for output parsing with automatic validation.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider


class ReviewResult(BaseModel):
    """Structured output: a code review verdict."""
    summary: str
    issues: list[str]
    severity: str
    approved: bool


provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Code Reviewer",
    goal="Review code snippets and return structured verdicts",
    instructions=(
        "Review the given code. Return your analysis as JSON with exactly these fields: "
        "summary (string), issues (list of strings), severity (low/medium/high/critical), approved (boolean)."
    ),
    response_model=ReviewResult,
)

result = asyncio.run(agent.arun(
    "Review this Python code:\n"
    "def get_user(id):\n"
    "    return db.execute(f'SELECT * FROM users WHERE id = {id}')"
))

review = result.structured
print(f"Summary: {review.summary}")
print(f"Severity: {review.severity}")
print(f"Approved: {review.approved}")
print(f"Issues: {review.issues}")
