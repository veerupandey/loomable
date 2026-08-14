"""Structured I/O — Validated input and output schemas.

USE WHEN: You need the agent to return structured data (JSON)
that conforms to a specific schema, not free-form text.

Uses Pydantic models for output parsing with automatic validation.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from pydantic import BaseModel

from loomable.agent import Agent


class ReviewResult(BaseModel):
    """Structured output: a code review verdict."""
    summary: str
    issues: list[str]
    severity: str
    approved: bool


provider = require_provider()

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
