"""Hello World Agent — The absolute minimum.

USE WHEN: You just want a single agent to answer a question.
This is the starting point for any loomable project.

One agent, one question, one answer — 3 lines of setup.
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

agent = Agent(
    model=provider,
    role="Helpful Assistant",
    goal="Answer questions clearly and concisely",
)

result = asyncio.run(agent.arun("What is the capital of France? Answer in one sentence."))
print(result.output.text())
