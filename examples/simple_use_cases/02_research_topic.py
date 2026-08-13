"""02 — Simple research: 'kageha matcha'

USE WHEN: You want an agent to look up an unfamiliar topic/product
and explain it clearly.

Question:
  Bring me info about kageha matcha (user spelling: kageha mactha).
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
    role="Product researcher",
    goal="Explain unfamiliar products or topics in plain English",
    instructions=(
        "Search the web for the topic. If spelling looks off, try close variants "
        "(e.g. matcha / Kagoshima matcha) and say what you searched. "
        "Explain what it is, where it comes from, and why people care. Be concise."
    ),
    tools=[WebSearchTools()],
)


async def main() -> None:
    question = "Bring me info about kageha mactha. What is it?"
    print("Q:", question, "\n")
    result = await agent.arun(question)
    print(result.output.text())
    if result.tool_activity:
        print(f"\n[tools used: {len(result.tool_activity)}]")


if __name__ == "__main__":
    asyncio.run(main())
