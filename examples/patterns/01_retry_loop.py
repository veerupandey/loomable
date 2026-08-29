"""Quality gates — Agent verifier and ``Workflow.loop``.

USE WHEN: You need verified output with automatic retry.

Two layers in this file:

- ``Agent(verifier=..., retry_on_failure=True)`` — single-agent gate
- ``Workflow(...).loop(agent, until=verify)`` — keep running a body until the
  check passes (or ``max_iterations``)

For a **bounded** generate → check → repair step inside a larger pipeline
(hard budget, optional ``reads=`` / ``on_failure``), use ``Workflow.verify``
instead — see ``patterns/07_graph_engineering.py``.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow
from loomable.display import pp

provider = require_provider()


def verify_has_code(output, ctx) -> bool:
    """Verify that the response contains a Python code block."""
    text = output.text()
    return "```python" in text or "def " in text


# --- Agent-level verifier ---
agent = Agent(
    model=provider,
    role="Python Developer",
    goal="Write correct Python code",
    instructions="Always include working Python code in your responses.",
    verifier=verify_has_code,
    retry_on_failure=True,
    max_verify_retries=2,
)

print("=== Agent verifier ===")
pp(asyncio.run(agent.arun("Write a function to check if a string is a palindrome.")))

# --- Workflow.loop gate ---
polisher = Agent(
    model=provider,
    role="Code Polisher",
    goal="Produce a short Python function with a code fence",
    instructions="Reply with a ```python code fence containing a working function.",
)

wf = Workflow("quality").loop(polisher, until=verify_has_code, max_iterations=3)
print("\n=== Workflow.loop ===")
print(asyncio.run(wf.arun("Write is_palindrome(s) in Python.")).output.text())
