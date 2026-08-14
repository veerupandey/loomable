"""Shared memory across a Workflow — Agents only.

Prefer ``Workflow(..., memory=True)`` (or ``Memory.compose`` on each Agent).
Prior Agent output is passed to the next Agent automatically.
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

wf = (
    Workflow("shared-memory-demo", memory=True)
    .step("research", researcher)
    .step("write", writer)
)

result = asyncio.run(wf.arun("What are the pros and cons of event sourcing?"))
print(result.output.text())
