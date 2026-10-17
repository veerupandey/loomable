"""Print text as it arrives and read the completed result."""

import asyncio

from examples._provider import make_provider
from loomable import Agent


async def main():
    async with make_provider() as model:
        agent = Agent(model)
        async for chunk in agent.astream("Explain a database index in two short sentences."):
            if chunk.delta.data:
                print(chunk.delta.data.decode(), end="", flush=True)
            if chunk.done:
                print(f"\nUsage {chunk.result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
