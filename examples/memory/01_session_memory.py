"""Recall an order reference after rebuilding an agent from saved history."""

import asyncio
from tempfile import TemporaryDirectory

from examples._provider import make_provider
from loomable import Agent, open_session_store


async def main():
    with TemporaryDirectory() as directory:
        async with make_provider() as model:
            store = open_session_store("file", path=directory)
            agent = Agent(model, session_id="ticket-42", session_store=store)
            await agent.arun("The order reference is QZ-8421. Acknowledge it.")
            reopened = Agent(model, session_id="ticket-42", session_store=store, resume=True)
            print((await reopened.arun("What was the order reference?")).text)


if __name__ == "__main__":
    asyncio.run(main())
