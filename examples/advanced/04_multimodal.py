"""Pass a local image to a model with image input support."""

import argparse
import asyncio
from pathlib import Path

from examples._provider import make_provider
from loomable import Agent


async def main(image: Path):
    async with make_provider() as model:
        agent = Agent(model, instructions="Describe only what is visible.")
        result = await agent.arun("Describe the main trend or notable feature in this image.", images=[image])
        print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path,
                        default=Path(__file__).with_name("fixtures") / "dashboard_spike.png")
    asyncio.run(main(parser.parse_args().image))
