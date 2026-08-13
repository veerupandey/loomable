"""01 — Simple Q&A: India news & Modi government

USE WHEN: You want a plain Agent that answers current-affairs questions
using web search tools.

Questions:
  1) What is the news in India?
  2) How is the Modi government doing?
"""

from __future__ import annotations


import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import asyncio

from loomable.agent import Agent
from loomable.toolkits import WebSearchTools

from _provider import make_provider

agent = Agent(
    model=make_provider(),
    role="India news analyst",
    goal="Give a clear, balanced briefing from recent public information",
    instructions=(
        "Use web_search for up-to-date facts. "
        "Answer in plain English with short bullets. "
        "Separate facts from opinion. Cite source titles when possible."
    ),
    tools=[WebSearchTools()],
)


async def main() -> None:
    question = (
        "What is the news in India right now, and how is the Modi government doing? "
        "Give a short briefing: top headlines, economy, and public sentiment."
    )
    print("Q:", question, "\n")
    result = await agent.arun(question)
    print(result.output.text())
    if result.tool_activity:
        print(f"\n[tools used: {len(result.tool_activity)}]")


if __name__ == "__main__":
    asyncio.run(main())
