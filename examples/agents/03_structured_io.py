"""Return a validated Pydantic object from a support message."""

import asyncio
from typing import Literal

from pydantic import BaseModel

from examples._provider import make_provider
from loomable import Agent


class Ticket(BaseModel):
    category: Literal["billing", "technical", "account"]
    summary: str


async def main():
    async with make_provider() as model:
        agent = Agent(model, instructions="Classify the support message.", response_model=Ticket)
        result = await agent.arun("I was charged twice for the same order.")
        print(result.structured.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
