"""Agent output chaining + Workflow blackboard for callable steps.

Prior Agent output is passed to the next Agent automatically via SharedState.
``Workflow(..., memory=True)`` attaches a blackboard on ``RunContext.memory``
for **callable** steps that ``write`` / ``recall`` it — Agent steps do not
consume the blackboard.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text

provider = require_provider()

researcher = Agent(
    model=provider,
    role="Researcher",
    goal="Research and store findings",
    instructions="Research the topic. Output clear key findings as short bullets.",
)

writer = Agent(
    model=provider,
    role="Writer",
    goal="Write based on the researcher's findings",
    instructions=(
        "You receive the researcher's findings as your input. "
        "Write a short balanced summary."
    ),
)


async def pin_findings(inp, *, context=None):
    """Callable step: persist prior Agent text on the Workflow blackboard."""
    text = inp.text() if callable(getattr(inp, "text", None)) else str(inp)
    memory = getattr(context, "memory", None) if context is not None else None
    if memory is not None:
        await memory.write(text)
    return RunResult(output=AgentOutput(parts=[Text(text)]), session_id="")


wf = (
    Workflow("shared-memory-demo", session_id="demo", memory=True)
    .step("research", researcher)
    .step("pin", pin_findings)
    .step("write", writer)
)

result = asyncio.run(wf.arun("What are the pros and cons of event sourcing?"))
print(result.output.text())
