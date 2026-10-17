"""Let a model call a typed, read-only inventory function."""

import asyncio

from examples._provider import make_provider
from loomable import Agent, tool

INVENTORY = {"WIDGET": 12, "SPROCKET": 0}


@tool
def stock_count(sku: str) -> int:
    """Return the number of units in the local inventory fixture."""
    return INVENTORY[sku.upper()]


async def main():
    async with make_provider() as model:
        agent = Agent(model, tools=[stock_count], require_tools=["stock_count"], strict_require_tools=True)
        result = await agent.arun("How many units of WIDGET are in stock?")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
