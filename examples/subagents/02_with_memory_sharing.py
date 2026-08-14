"""Subagents sharing a conversation session.

USE WHEN: Parent and specialists should see the same L1/L2 turns.

Give them the same ``session_id`` (or a shared ``Memory.compose`` bundle).
Delegation does not copy session context by itself.
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

researcher = Agent(
    model=provider,
    role="Researcher",
    goal="Find accurate technical information",
    instructions="Provide factual, detailed research results.",
    session_id="shared-context",
)

writer = Agent(
    model=provider,
    role="Technical Writer",
    goal="Write clear documentation from research",
    instructions="Write concise, well-structured documentation.",
    session_id="shared-context",
)

lead = Agent(
    model=provider,
    role="Documentation Lead",
    goal="Produce high-quality technical documentation",
    instructions=(
        "First delegate research, then delegate writing based on the research. "
        "Return the final documentation."
    ),
    subagents=[researcher, writer],
    session_id="shared-context",
)

result = asyncio.run(lead.arun(
    "Create documentation for a REST API rate limiter middleware."
))
print(result.output.text())
