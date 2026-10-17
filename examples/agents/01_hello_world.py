"""Give an agent instructions, ask one question, and read the answer."""

import asyncio

from examples._provider import make_provider
from loomable import Agent


async def main():
    async with make_provider() as model:
        agent = Agent(model=model, instructions="Answer in one clear sentence.")
        result = await agent.arun("What is a database index?")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
