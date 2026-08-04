"""Retry Loop — Quality gate with automatic retry.

USE WHEN: You need verified output quality. The agent retries
until a verifier function approves the result.

Combines Agent + Loop + Verifier for self-correcting behavior.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()


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
