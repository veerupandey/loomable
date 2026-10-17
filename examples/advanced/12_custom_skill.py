"""Load a custom skill and call its local script tool."""

import asyncio
from pathlib import Path

from examples._provider import make_provider
from loomable import Agent


async def main():
    skill = Path(__file__).resolve().parents[1] / "skills" / "weather-lookup"
    async with make_provider() as model:
        agent = Agent(
            model, skills=[str(skill)], require_tools=["get_weather"], strict_require_tools=True,
        )
        print((await agent.arun("What is the historical sample temperature for Vancouver?")).text)


if __name__ == "__main__":
    asyncio.run(main())
