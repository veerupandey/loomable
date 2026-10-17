"""Retry transient model failures and replay a completed job by request ID."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_provider
from loomable import Agent, SQLiteCheckpointer
from loomable.agent import DurabilityOptions
from loomable.providers import RetryPolicy


async def main():
    with TemporaryDirectory() as directory:
        store = SQLiteCheckpointer(str(Path(directory) / "jobs.sqlite"))
        async with make_provider() as model:
            def worker():
                return Agent(
                    model, session_id="ticket-42-summary",
                    durability=DurabilityOptions(store), resilience=RetryPolicy(max_attempts=3),
                )

            prompt = "Summarize in one sentence: the customer was charged twice for order QZ-8421."
            first = await worker().arun_once(prompt)
            replay = await worker().arun_once(prompt)
            print(first.text)
            print(f"Replayed {replay.metadata['replayed']}; run ID {replay.metadata['run_id']}")


if __name__ == "__main__":
    asyncio.run(main())
