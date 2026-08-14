"""Agent verifier retry — Quality gate with automatic retry.

USE WHEN: You need verified output quality. The agent retries
until a verifier function approves the result.

This demo uses Agent-level ``verifier=`` + ``retry_on_failure=True``.
For a Flow-level quality gate, use ``Loop`` from ``loomable.flow``::

    from loomable.flow import Loop
    loop = Loop(agent, verifier=verify_has_code, max_iterations=3)
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent

provider = require_provider()


def verify_has_code(output, ctx) -> bool:
    """Verify that the response contains a Python code block."""
    text = output.text()
    return "```python" in text or "def " in text


agent = Agent(
    model=provider,
    role="Python Developer",
    goal="Write correct Python code",
    instructions="Always include working Python code in your responses.",
    verifier=verify_has_code,
    retry_on_failure=True,
    max_verify_retries=2,
)

result = asyncio.run(agent.arun("Write a function to check if a string is a palindrome."))

# Pretty-print with loop/verification info
from loomable.display import pp

pp(result)
