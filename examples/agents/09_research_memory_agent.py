"""Run a research assistant that remembers preferences across turns."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_embedder, make_provider
from examples.agents.research_memory_agent import build_research_agent


async def main(pdfs: list[Path], question: str | None):
    with TemporaryDirectory() as directory:
        async with make_provider() as model:
            agent = build_research_agent(
                model, embedder=make_embedder(), workspace=directory,
                session_id="research-demo", user_id="sam", pdf_paths=pdfs,
            )
            await agent.arun("My name is Sam. I prefer bulleted research summaries.")
            print((await agent.arun("What do you remember about my writing preferences?")).text)
            if question:
                print((await agent.arun(question)).text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", default=[])
    parser.add_argument("--question", help="Optional research question for the specialist team.")
    args = parser.parse_args()
    asyncio.run(main(args.pdf, args.question))
