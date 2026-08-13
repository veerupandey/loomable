"""05 — Tool calling smoke test

USE WHEN: You want to verify the agent actually calls tools
(not just answers from model memory).
"""

from __future__ import annotations


import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import asyncio

from loomable.agent import Agent, tool
from loomable.display import pp

from _provider import make_provider


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def city_fact(city: str) -> str:
    """Return a canned fact about a city (demo tool)."""
    facts = {
        "delhi": "Delhi is the capital territory of India.",
        "mumbai": "Mumbai is India's financial hub.",
        "bengaluru": "Bengaluru is a major tech center in India.",
    }
    return facts.get(city.strip().lower(), f"No canned fact for {city}.")


agent = Agent(
    model=make_provider(),
    role="Tool-using assistant",
    goal="Always use tools for math and city facts",
    instructions=(
        "You MUST use tools for arithmetic and city_fact lookups. "
        "Do not invent tool results."
    ),
    tools=[add, multiply, city_fact],
)


async def main() -> None:
    result = await agent.arun(
        "What is (12 + 8) * 3? Also give me a city_fact for Delhi."
    )
    pp(result)
    assert result.tool_activity, "Expected at least one tool call"
    print(f"\nTool calls: {len(result.tool_activity)}")
    for activity in result.tool_activity:
        name = getattr(activity, "tool_name", None) or getattr(
            getattr(activity, "result", None), "tool_name", "?"
        )
        print(f"  - {name}: {activity.result.content}")


if __name__ == "__main__":
    asyncio.run(main())
