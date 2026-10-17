"""Run a release review with named specialists. Pick who controls delegation."""

import argparse
import asyncio

from examples._provider import make_provider
from loomable import Agent, Team


async def main(mode: str):
    async with make_provider() as model:
        database = Agent(
            model, role="Database reviewer",
            instructions="Identify one migration risk and propose a mitigation.",
        )
        rollout = Agent(
            model, role="Rollout reviewer",
            instructions="Identify one deployment risk and propose a mitigation.",
        )
        team = Team([database, rollout], model=model, mode=mode)
        result = await team.arun("Checkout will add a non-null column to its orders table. Review the change.")
        print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["route", "coordinate", "broadcast", "sequential", "tasks"],
                        default="route")
    asyncio.run(main(parser.parse_args().mode))
